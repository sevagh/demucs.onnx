#include "demucs.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <map>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stddef.h>
#include <tuple>
#include <vector>

using namespace nqr;

// Overload for file path, calling one of the other overloads as needed
static demucsonnx::demucs_model load_model(
    const std::string& htdemucs_model_path,
    Ort::SessionOptions& session_options
) {
    struct demucsonnx::demucs_model model;

    std::ifstream file(htdemucs_model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open model file: " + htdemucs_model_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> file_data(size);
    if (!file.read(file_data.data(), size)) {
        throw std::runtime_error("Failed to read model file.");
    }

    bool success = demucsonnx::load_model(file_data, model, session_options);
    if (!success) {
        throw std::runtime_error("Failed to load model.");
    }

    return model;
}

static Eigen::MatrixXf load_audio_file(std::string filename)
{
    // load a wav file with libnyquist
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    NyquistIO loader;

    loader.Load(fileData.get(), filename);

    if (fileData->sampleRate != demucsonnx::SUPPORTED_SAMPLE_RATE)
    {
        std::cerr << "[ERROR] demucs.cpp only supports the following sample "
                     "rate (Hz): "
                  << demucsonnx::SUPPORTED_SAMPLE_RATE << std::endl;
        exit(1);
    }

    std::cout << "Input samples: "
              << fileData->samples.size() / fileData->channelCount << std::endl;
    std::cout << "Length in seconds: " << fileData->lengthSeconds << std::endl;
    std::cout << "Number of channels: " << fileData->channelCount << std::endl;

    if (fileData->channelCount != 2 && fileData->channelCount != 1)
    {
        std::cerr << "[ERROR] demucs.cpp only supports mono and stereo audio"
                  << std::endl;
        exit(1);
    }

    // number of samples per channel
    std::size_t N = fileData->samples.size() / fileData->channelCount;

    // create a struct to hold two float vectors for left and right channels
    Eigen::MatrixXf ret(2, N);

    if (fileData->channelCount == 1)
    {
        // Mono case
        for (std::size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[i]; // left channel
            ret(1, i) = fileData->samples[i]; // right channel
        }
    }
    else
    {
        // Stereo case
        for (std::size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[2 * i];     // left channel
            ret(1, i) = fileData->samples[2 * i + 1]; // right channel
        }
    }

    return ret;
}

// write a function to write a StereoWaveform to a wav file
static bool write_audio_file(const Eigen::MatrixXf &waveform,
                             std::string filename)
{
    // create a struct to hold the audio data
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    // set the sample rate
    fileData->sampleRate = demucsonnx::SUPPORTED_SAMPLE_RATE;

    // set the number of channels
    fileData->channelCount = 2;

    // set the number of samples
    fileData->samples.resize(waveform.cols() * 2);

    // write the left channel
    for (long int i = 0; i < waveform.cols(); ++i)
    {
        fileData->samples[2 * i] = waveform(0, i);
        fileData->samples[2 * i + 1] = waveform(1, i);
    }

    int encoderStatus =
        encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE},
                           fileData.get(), filename);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
    return encoderStatus == 0;
}

int main(int argc, const char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model file> <wav file> <out dir>"
                  << std::endl;
        exit(1);
    }

    std::cout << "demucs.onnx Main driver program" << std::endl;
    std::string model_file = argv[1];

    // load audio passed as argument
    std::string wav_file = argv[2];

    // strip extension to make prefix for output filenames
    std::filesystem::path output_file_prefix = std::filesystem::path(wav_file).stem();

    // output dir passed as argument
    std::string out_dir = argv[3];

    // Check if the output directory exists, and create it if not
    std::filesystem::path output_dir_path(out_dir);
    if (!std::filesystem::exists(output_dir_path))
    {
        std::cerr << "Directory does not exist: " << out_dir << ". Creating it."
                  << std::endl;
        if (!std::filesystem::create_directories(output_dir_path))
        {
            std::cerr << "Error: Unable to create directory: " << out_dir
                      << std::endl;
            return 1;
        }
    }
    else if (!std::filesystem::is_directory(output_dir_path))
    {
        std::cerr << "Error: " << out_dir << " exists but is not a directory!"
                  << std::endl;
        return 1;
    }

    Eigen::MatrixXf audio = load_audio_file(wav_file);
    Eigen::Tensor3dXf out_targets;

    std::cout << "Running Demucs.onnx inference for: " << wav_file << std::endl;

        // set output precision to 3 decimal places
    std::cout << std::fixed << std::setprecision(3);

    demucsonnx::ProgressCallback progressCallback =
        [](float progress, const std::string &log_message)
    {
        std::cout << "(" << std::setw(3) << std::setfill(' ')
                  << progress * 100.0f << "%) " << log_message << std::endl;
    };

    // create Ort::SessionOptions
    Ort::SessionOptions session_options;

    // max out threads and increase performance to the max on my beefy
    // desktop CPU
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.SetIntraOpNumThreads(0);
    session_options.SetInterOpNumThreads(1);

    // General optimizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    struct demucsonnx::demucs_model model = load_model(
        model_file,
        session_options
    );

    auto preProcessingTime = std::chrono::high_resolution_clock::now();

    // create 4 audio matrix same size, to hold output
    Eigen::Tensor3dXf audio_targets =
        demucsonnx::demucs_inference(model, audio, progressCallback);

    auto postProcessingTime = std::chrono::high_resolution_clock::now();
    auto processingTime =
            std::chrono::duration_cast<std::chrono::milliseconds>(postProcessingTime - preProcessingTime);
    std::cout << "Stems split in " << std::format("{:%M:%S}", processingTime) << "s" << std::endl;

    out_targets = audio_targets;

    int nb_out_sources = model.nb_sources;

    bool writeOk = true;
    for (int target = 0; target < nb_out_sources; ++target)
    {
        // now write the 4 audio waveforms to files in the output dir
        // using libnyquist
        // join out_dir with "/target_0.wav"
        // using std::filesystem::path;

        std::filesystem::path p = out_dir;
        // make sure the directory exists
        std::filesystem::create_directories(p);

        auto p_target = p / "target_0.wav";

        // target 0,1,2,3 map to drums,bass,other,vocals

        std::string target_name;

        switch (target)
        {
        case 0:
            target_name = "drums";
            break;
        case 1:
            target_name = "bass";
            break;
        case 2:
            target_name = "other";
            break;
        case 3:
            target_name = "vocals";
            break;
        case 4:
            target_name = "guitar";
            break;
        case 5:
            target_name = "piano";
            break;
        default:
            std::cerr << "Error: target " << target << " not supported"
                      << std::endl;
            exit(1);
        }

        // insert target_name into the path after the digit
        // e.g. target_name_0_drums.wav
        p_target.replace_filename(output_file_prefix.filename().string() + "_" + std::to_string(target) + "_" +
                                  target_name + ".wav");

        std::cout << "Writing wav file " << p_target << std::endl;

        Eigen::MatrixXf target_waveform(2, audio.cols());

        // copy the input stereo wav file into all 4 targets
        for (int channel = 0; channel < 2; ++channel)
        {
            for (int sample = 0; sample < audio.cols(); ++sample)
            {
                target_waveform(channel, sample) =
                    out_targets(target, channel, sample);
            }
        }

        writeOk = write_audio_file(target_waveform, p_target) && writeOk;
    }

    return writeOk ? 0 : 1;
}
