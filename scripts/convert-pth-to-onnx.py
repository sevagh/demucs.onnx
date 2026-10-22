#!/usr/bin/env python

import sys
import torch
from torch.nn import functional as F
import argparse
from pathlib import Path
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs, standalone_spec, standalone_magnitude

DEMUCS_MODEL = "htdemucs"
DEMUCS_MODEL_6S = "htdemucs_6s"
DEMUCS_MODEL_FT = "htdemucs_ft"
DEMUCS_MODEL_FT_DRUMS = "htdemucs_ft_drums"
DEMUCS_MODEL_FT_BASS = "htdemucs_ft_bass"
DEMUCS_MODEL_FT_OTHER = "htdemucs_ft_other"
DEMUCS_MODEL_FT_VOCALS = "htdemucs_ft_vocals"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Demucs PyTorch models to ONNX')
    parser.add_argument("dest_dir", type=str, help="destination path for the converted model")
    parser.add_argument("--six-source", default=False, action="store_true", help="convert 6s model (default: 4s)")
    parser.add_argument("--ft-drums", default=False, action="store_true", help="convert fine-tuned drum model")
    parser.add_argument("--ft-bass", default=False, action="store_true", help="convert fine-tuned bass model")
    parser.add_argument("--ft-other", default=False, action="store_true", help="convert fine-tuned other model")
    parser.add_argument("--ft-vocals", default=False, action="store_true", help="convert fine-tuned vocals model")

    args = parser.parse_args()

    dir_out = Path(args.dest_dir)
    dir_out.mkdir(parents=True, exist_ok=True)

    # Load the appropriate model
    model = get_model(DEMUCS_MODEL)
    model_name = DEMUCS_MODEL
    if args.six_source:
        model = get_model(DEMUCS_MODEL_6S)
        model_name = DEMUCS_MODEL_6S
    elif args.ft_drums:
        model = get_model(DEMUCS_MODEL_FT_DRUMS)
        model_name = DEMUCS_MODEL_FT_DRUMS
    elif args.ft_bass:
        model = get_model(DEMUCS_MODEL_FT_BASS)
        model_name = DEMUCS_MODEL_FT_BASS
    elif args.ft_other:
        model = get_model(DEMUCS_MODEL_FT_OTHER)
        model_name = DEMUCS_MODEL_FT_OTHER
    elif args.ft_vocals:
        model = get_model(DEMUCS_MODEL_FT_VOCALS)
        model_name = DEMUCS_MODEL_FT_VOCALS

    # Check if model is an instance of BagOfModels
    if isinstance(model, HTDemucs):
        core_model = model
    elif hasattr(model, 'models') and isinstance(model.models[0], HTDemucs):
        core_model = model.models[0]  # Select the first model in BagOfModels
    else:
        raise TypeError("Unsupported model type")

    # Prepare a dummy input tensor
    dummy_waveform = torch.randn(1, 2, 343980)

    # pre-pad the dummy_waveform since we removed padding from NN itself
    #        training_length = int(self.segment * self.samplerate)
    #        if mix.shape[-1] < training_length:
    #            length_pre_pad = mix.shape[-1]
    #            mix = F.pad(mix, (0, training_length - length_pre_pad))

    training_length = int(core_model.segment * core_model.samplerate)
    dummy_waveform = F.pad(dummy_waveform, (0, training_length - dummy_waveform.shape[-1]))

    magspec = standalone_magnitude(standalone_spec(dummy_waveform))

    dummy_input = (dummy_waveform, magspec)

    # Define output file name
    onnx_file_path = dir_out / f"{model_name}.onnx"

    # Export the core model to ONNX
    try:
        torch.onnx.export(
            core_model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # torch>=2.9 defaults to the dynamo exporter, which produces a
            # different graph (and requires onnxscript). The C++ inference code
            # in src/ expects the TorchScript exporter graph on opset 17, so
            # pin the legacy exporter explicitly.
            dynamo=False,
        )
        print(f"Model successfully converted to ONNX format at {onnx_file_path}")
    except Exception as e:
        print("Error during ONNX export:", e)
