#include "demucs.hpp"
#include "dsp.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <Eigen/Dense>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "demucs.hpp"
// this is the model model baked into a header file
#include "htdemucs.ort.h"

static std::tuple<int, int>
symmetric_zero_padding(Eigen::MatrixXf &padded, const Eigen::MatrixXf &original,
                       int total_padding)
{
    int left_padding = std::floor((float)total_padding / 2.0f);
    int right_padding = total_padding - left_padding;

    int N = original.cols(); // The original number of columns

    // Copy the original mix into the middle of padded_mix
    padded.block(0, left_padding, 2, N) = original;

    // Zero padding on the left
    padded.block(0, 0, 2, left_padding) =
        Eigen::MatrixXf::Zero(2, left_padding);

    // Zero padding on the right
    padded.block(0, N + left_padding, 2, right_padding) =
        Eigen::MatrixXf::Zero(2, right_padding);

    // return left, right padding as tuple
    return std::make_tuple(left_padding, right_padding);
}

// forward declaration of inner fns
static Eigen::Tensor3dXf
shift_inference(Ort::Session &model,
                Eigen::MatrixXf &full_audio, demucsonnx::ProgressCallback cb);

static Eigen::Tensor3dXf
split_inference(Ort::Session &model,
                Eigen::MatrixXf &full_audio, demucsonnx::ProgressCallback cb);

static Eigen::Tensor3dXf segment_inference(
    Ort::Session &model, Eigen::MatrixXf chunk,
    int segment_sample, struct demucsonnx::demucs_segment_buffers &buffers,
    struct demucsonnx::stft_buffers &stft_buf, demucsonnx::ProgressCallback cb,
    float current_progress, float segment_progress);

Eigen::Tensor3dXf demucsonnx::demucs_inference(Ort::Session &model,
                                              const Eigen::MatrixXf &audio,
                                              demucsonnx::ProgressCallback cb)
{
    // working copy to modify
    Eigen::MatrixXf full_audio = audio;

    // first, normalize the audio to mean and std
    // ref = wav.mean(0)
    // wav = (wav - ref.mean()) / ref.std()
    // Calculate the overall mean and standard deviation
    // Compute the mean and standard deviation separately for each channel
    Eigen::VectorXf ref_mean_0 = full_audio.colwise().mean();

    float ref_mean = ref_mean_0.mean();
    float ref_std = std::sqrt((ref_mean_0.array() - ref_mean).square().sum() /
                              (ref_mean_0.size() - 1));

    // Normalize the audio
    Eigen::MatrixXf normalized_audio =
        (full_audio.array() - ref_mean) / ref_std;

    full_audio = normalized_audio;

    Eigen::Tensor3dXf waveform_outputs = shift_inference(model, full_audio, cb);

    // now inverse the normalization in Eigen C++
    // sources = sources * ref.std() + ref.mean()
    waveform_outputs = (waveform_outputs * ref_std).eval() + ref_mean;

    return waveform_outputs;
}

static Eigen::Tensor3dXf
shift_inference(Ort::Session &model,
                Eigen::MatrixXf &full_audio, demucsonnx::ProgressCallback cb)
{
    // first, apply shifts for time invariance
    // we simply only support shift=1, the demucs default
    // shifts (int): if > 0, will shift in time `mix` by a random amount between
    // 0 and 0.5 sec
    //     and apply the oppositve shift to the output. This is repeated
    //     `shifts` time and all predictions are averaged. This effectively
    //     makes the model time equivariant and improves SDR by up to 0.2
    //     points.
    int max_shift =
        (int)(demucsonnx::MAX_SHIFT_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);

    int length = full_audio.cols();

    Eigen::MatrixXf padded_mix(2, length + 2 * max_shift);

    symmetric_zero_padding(padded_mix, full_audio, 2 * max_shift);

    int offset = rand() % max_shift;
    // int offset = 1337;

    std::cout << "1., apply model w/ shift, offset: " << offset << std::endl;

    Eigen::MatrixXf shifted_audio =
        padded_mix.block(0, offset, 2, length + max_shift - offset);

    Eigen::Tensor3dXf waveform_outputs =
        split_inference(model, shifted_audio, cb);

    const int nb_out_sources = 4;

    // trim the output to the original length
    // waveform_outputs = waveform_outputs[..., max_shift:max_shift + length]
    Eigen::Tensor3dXf trimmed_waveform_outputs =
        waveform_outputs
            .reshape(Eigen::array<int, 3>(
                {nb_out_sources, 2,
                 static_cast<int>(waveform_outputs.dimension(2))}))
            .slice(Eigen::array<int, 3>({0, 0, max_shift - offset}),
                   Eigen::array<int, 3>({nb_out_sources, 2, length}));

    return trimmed_waveform_outputs;
}

static Eigen::Tensor3dXf
split_inference(Ort::Session &model,
                Eigen::MatrixXf &full_audio, demucsonnx::ProgressCallback cb)
{
    // calculate segment in samples
    int segment_samples =
        (int)(demucsonnx::SEGMENT_LEN_SECS * demucsonnx::SUPPORTED_SAMPLE_RATE);

    const int nb_out_sources = 4;

    // let's create reusable buffers with padded sizes
    struct demucsonnx::demucs_segment_buffers buffers(2, segment_samples,
                                                     nb_out_sources);
    struct demucsonnx::stft_buffers stft_buf(buffers.padded_segment_samples);

    // next, use splits with weighted transition and overlap
    // split (bool): if True, the input will be broken down in 8 seconds
    // extracts
    //     and predictions will be performed individually on each and
    //     concatenated. Useful for model with large memory footprint like
    //     Tasnet.

    int stride_samples = (int)((1 - demucsonnx::OVERLAP) * segment_samples);

    int length = full_audio.cols();

    // create an output tensor of zeros for four source waveforms
    Eigen::Tensor3dXf out = Eigen::Tensor3dXf(nb_out_sources, 2, length);
    out.setZero();

    // create weight tensor
    Eigen::VectorXf weight(segment_samples);
    weight.setZero();

    weight.head(segment_samples / 2) =
        Eigen::VectorXf::LinSpaced(segment_samples / 2, 1, segment_samples / 2);
    weight.tail(segment_samples / 2) =
        weight.head(segment_samples / 2).reverse();
    weight /= weight.maxCoeff();
    weight = weight.array().pow(demucsonnx::TRANSITION_POWER);

    Eigen::VectorXf sum_weight(length);
    sum_weight.setZero();

    // i prefer using `std::ceilf` but :shrug:
    int total_chunks = ::ceilf((float)length / (float)stride_samples);
    float increment_per_chunk = 1.0f / (float)total_chunks;
    float inference_progress = 0.0f;

    for (int offset = 0; offset < length; offset += stride_samples)
    {
        // create a chunk of the padded_full_audio
        int chunk_end = std::min(segment_samples, length - offset);
        Eigen::MatrixXf chunk = full_audio.block(0, offset, 2, chunk_end);
        int chunk_length = chunk.cols();

        std::cout << "2., apply model w/ split, offset: " << offset
                  << ", chunk shape: (" << chunk.rows() << ", " << chunk.cols()
                  << ")" << std::endl;

        Eigen::Tensor3dXf chunk_out =
            segment_inference(model, chunk, segment_samples, buffers, stft_buf,
                              cb, inference_progress, increment_per_chunk);

        // add the weighted chunk to the output
        // out[..., offset:offset + segment] += (weight[:chunk_length] *
        // chunk_out).to(mix.device)
        for (int i = 0; i < nb_out_sources; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int k = 0; k < chunk_length; ++k)
                {
                    if (offset + k >= length)
                    {
                        break;
                    }
                    out(i, j, offset + k) +=
                        weight(k % chunk_length) * chunk_out(i, j, k);
                }
            }
        }

        // sum_weight[offset:offset + segment] +=
        // weight[:chunk_length].to(mix.device)
        for (int k = 0; k < chunk_length; ++k)
        {
            if (offset + k >= length)
            {
                break;
            }
            sum_weight(offset + k) += weight(k % chunk_length);
        }

        inference_progress += increment_per_chunk;
    }

    for (int i = 0; i < nb_out_sources; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < length; ++k)
            {
                out(i, j, k) /= sum_weight[k];
            }
        }
    }
    return out;
}

static Eigen::Tensor3dXf segment_inference(
    Ort::Session &model, Eigen::MatrixXf chunk,
    int segment_samples, struct demucsonnx::demucs_segment_buffers &buffers,
    struct demucsonnx::stft_buffers &stft_buf, demucsonnx::ProgressCallback cb,
    float current_progress, float segment_progress)
{
    int chunk_length = chunk.cols();

    // copy chunk into buffers.mix with symmetric zero-padding
    // assign two ints to tuple return value
    std::tuple<int, int> padding = symmetric_zero_padding(
        buffers.mix, chunk, segment_samples - chunk_length);

    // apply demucs inference via ONNX
    demucsonnx::model_inference(model, buffers, stft_buf, cb, current_progress,
                               segment_progress);

    const int nb_out_sources = 4;

    // copy from buffers.targets_out into chunk_out with center trimming
    Eigen::Tensor3dXf chunk_out =
        Eigen::Tensor3dXf(nb_out_sources, 2, chunk_length);
    chunk_out.setZero();

    for (int i = 0; i < nb_out_sources; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < chunk_length; ++k)
            {
                // undoing center_trim
                chunk_out(i, j, k) =
                    buffers.targets_out(i, j, k + std::get<0>(padding));
            }
        }
    }

    return chunk_out;
}
