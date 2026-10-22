# demucs.onnx

> **This is the `demucs.wrapped` fork** of [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx):
> a Windows/MSVC build with selectable execution providers (CPU / CUDA / DirectML),
> plus sample-rate conversion tooling. Everything fork-specific — how to build, the
> GPU setup, measured numbers, and what to watch out for — lives in **[WINDOWS.md](WINDOWS.md)**.
> The rest of this README is upstream's.

ONNX inference for the [Demucs v4 hybrid transformer](https://github.com/facebookresearch/demucs) model, a high-performance and high quality PyTorch neural network for music source separation.

Unlike my other project [demucs.cpp](https://github.com/sevagh/demucs.cpp), where I wrote all of the neural network layers by hand, in this repo I use [ONNXRuntime](https://github.com/microsoft/onnxruntime), which is a universal high-performance neural network inference library with native support for diverse hardware platforms (CPU, GPU, desktop OS, web, smartphone OS, etc.).

This code should perform significantly faster than my other project, and you can also leverage the different ONNX execution providers e.g. CUDA, GPUs, WebGPU, WebNN.

## Idea and Python/PyTorch implementation

The original Demucs v4 does the STFT and iSTFT (spectrogram and inverse spectrogram) inside the neural network, and these operations are not exportable to ONNX, so this basic export will usually fail:
```
torch.onnx.export(demucs_model, opset_version=17, ...)
```

I copy demucs into [demucs-for-onnx](./demucs-for-onnx), and simply move the stft/istft _outside_ of the network itself in `htdemucs.py`, such that the core model expects and returns spectrograms:
```
# HTDemucs class methods moved to standalone functions
def standalone_spec(x, nfft=4096, hop_length=4096//4):
def standalone_magnitude(z, cac=True):
def standalone_ispec(z, length=None, scale=0, hop_length=4096//4):
def standalone_mask(z, m, cac=True, wiener_iters=0, training=False, end_iters=0):

class HTDemucs:
    # inputs (mix, x) = time-domain waveform and complex-as-channels spectrogram
    # skip stft/istft in the network itself
    def forward(self, mix, x):
        ...
        return x, xt
```
In `apply.py`, I apply the standalone variants of the spec/ispec/mag/mask class methods to call them outside of the model:
```
# now that we chopped up demucs to remove the stft/istft
# from the model itself, we need to do that before and after inference
with th.no_grad():
    training_length = int(model.segment * model.samplerate)
    # this is the padding previously done in the model
    padded_padded_mix = F.pad(padded_mix, (0, training_length - padded_mix.shape[-1]))
    magspec = standalone_magnitude(standalone_spec(padded_padded_mix))
    out_x, out_xt = model(padded_mix, magspec)  # core model call
    out = out_xt + out
    out = out[..., :valid_length]
```

## C++ implementation

In C++, I borrow the STFT, iSTFT (with complex-as-channels) and padding functions from demucs.cpp and use them before calling the ONNX demucs in `src/model_inference.cpp`:
```
// run core demucs inference using onnx
void demucsonnx::model_inference(
    struct demucsonnx::demucs_model &model,
    struct demucsonnx::demucs_segment_buffers &buffers,
    struct demucsonnx::stft_buffers &stft_buf)
{
    // let's get a stereo complex spectrogram first
    demucsonnx::stft(stft_buf, buffers.padded_mix, buffers.z);

    // prepare frequency branch input by copying buffers.z into input_tensors[1]
    // to create x ('magnitude' spectrogram with complex-as-channels)
    float *x_onnx_data = buffers.input_tensors[1].GetTensorMutableData<float>();
    ...

    // prepare time branch input by copying buffers.mix into  input_tensors[0]
    float *xt_onnx_data = buffers.input_tensors[0].GetTensorMutableData<float>();

    // now we have the stft, apply the core demucs inference
    // (where we removed the stft/istft to successfully convert to ONNX)
    RunONNXInference(model, buffers);

    // Run the model
    model.sess->Run(
        demucsonnx::run_options,
        model.input_names_ptrs.data(),
        buffers.input_tensors.data(),
        buffers.input_tensors.size(),
        model.output_names_ptrs.data(),
        buffers.output_tensors.data(),
        model.output_names_ptrs.size()
    );

    std::cout << "ONNX inference completed." << std::endl;
```

## Instructions
First clone this repo with submodules to get all vendored libraries (onnxruntime, Eigen, etc.):
```
$ git clone --recurse-submodules https://github.com/sevagh/demucs.onnx
```

Install standard C++ dependencies, e.g. CMake, gcc, C++/g++, OpenBLAS for your OS (my instructions are for Pop!\_OS 22.04).

Also, set up an isolated Python environment with your tool of choice (I like [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)) and install the `scripts/requirements.txt` file:
```
$ mamba create --name demucsonnx python=3.12
$ mamba activate demucsonn
$ python -m pip install -r ./scripts/requirements.txt
```

### Convert PyTorch model to ONNX and ORT

Convert Demucs PyTorch model to ONNX:
```
$ python ./scripts/convert-pth-to-onnx.py ./demucs-onnx
...
Model successfully converted to ONNX format at onnx-models/htdemucs.onnx
```

You can convert the 4-source, 6-source, and fine-tuned models. Then, convert ONNX to ORT:
```
$ ./scripts/convert-model-to-ort.sh ./demucs-onnx
...
Converting optimized ONNX model /home/sevagh/repos/demucs.onnx/onnx-models/htdemucs.onnx to ORT format model /home/sevagh/repos/demucs.onnx/onnx-models/tmpmp673xjb.without_runtime_opt/htdemucs.ort
Converted 1/1 models successfully.
Generating config file from ORT format models with optimization style 'Runtime' and level 'all'
2024-11-11 08:10:05,695 ort_format_model.utils [INFO] - Created config in /home/sevagh/repos/demucs.onnx/onnx-models/htdemucs.required_operators_and_types.with_runtime_opt.config
```

### Build ONNXRuntime

Using the [ort-builder](https://github.com/olilarkin/ort-builder) strategy, we build a minimal onnxruntime library that only includes the specific types and operators needed for Demucs. I only provide a Linux build script (`./scripts/build-ort-linux.sh`).

Then, the CMakeLists.txt file for this application's sample CLI code (in `src_cli`) links against this built onnxruntime library.

### Build this C++ code

After building ONNXRuntime, compile with CMake (through a convenience target defined in the top-level Makefile:
```
$ make cli
$ ./build/build-cli/demucs ./onnx-models/htdemucs.ort ~/Music/unas.wav ./demucs-onnx-out
demucs.onnx Main driver program
Input samples: 2646000
Length in seconds: 60
Number of channels: 2
Running Demucs.onnx inference for: /home/sevagh/Music/unas.wav
shift offset is: 3062
ONNX inference completed.
(9.091%) Segment inference complete
...
ONNX inference completed.
(100.000%) Segment inference complete
Writing wav file "./demucs-onnx-out/target_0_drums.wav"
Encoder Status: 0
Writing wav file "./demucs-onnx-out/target_1_bass.wav"
Encoder Status: 0
Writing wav file "./demucs-onnx-out/target_2_other.wav"
Encoder Status: 0
Writing wav file "./demucs-onnx-out/target_3_vocals.wav"
Encoder Status: 0
```
