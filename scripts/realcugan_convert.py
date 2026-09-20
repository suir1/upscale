"""Convert a Real-CUGAN 2x PyTorch checkpoint to a fixed-shape Core ML model.

This is an experimental local-only converter.  The Real-CUGAN source tree is
passed separately because the upstream model implementation is not vendored in
this repository.
"""

import argparse
import importlib.util
import os

import coremltools as ct
import numpy as np
import torch


def replace_negative_padding():
    """Make the crop-style pads in the upstream model Core ML compatible."""

    original_pad = torch.nn.functional.pad

    def pad(input_tensor, pad, mode="constant", value=None):
        if any(amount < 0 for amount in pad):
            slices = [slice(None)] * input_tensor.dim()
            dimensions = len(pad) // 2
            for index in range(dimensions):
                left = pad[index * 2]
                right = pad[index * 2 + 1]
                dimension = input_tensor.dim() - 1 - index
                start = -left if left < 0 else None
                end = right if right < 0 else None
                slices[dimension] = slice(start, end)
            input_tensor = input_tensor[tuple(slices)]
            pad = tuple(max(0, amount) for amount in pad)
        if any(amount > 0 for amount in pad):
            return original_pad(input_tensor, pad, mode=mode, value=value)
        return input_tensor

    torch.nn.functional.pad = pad


class CoreMLUpCunet2x(torch.nn.Module):
    """Expose the non-tiled 2x path with a float input/output contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Aidoku supplies a fixed even-sized tile, so the dynamic padding and
        # odd-size crop branches in the upstream image wrapper are unnecessary.
        # Its shared multiarray decoder adds half of one 8-bit step to avoid
        # clipping in Real-ESRGAN; undo that offset for Real-CUGAN.
        x = torch.clamp(x - 0.00196078411, 0, 1)
        x = torch.nn.functional.pad(x, (18, 18, 18, 18), mode="reflect")
        x = self.model.unet1.forward(x)
        residual = self.model.unet2.forward(x, 1.0)
        x = x[:, :, 20:-20, 20:-20]
        return residual + x


def load_module(path):
    spec = importlib.util.spec_from_file_location("realcugan_upcunet", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Real-CUGAN source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Real-CUGAN 2x checkpoint to Core ML"
    )
    parser.add_argument("pth_path", help="Path to up2x-latest-*.pth")
    parser.add_argument(
        "--source-file",
        required=True,
        help="Path to the upstream Real-CUGAN upcunet_v3.py file",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--output", help="Output .mlpackage path")
    args = parser.parse_args()

    replace_negative_padding()
    source = load_module(args.source_file)
    checkpoint = torch.load(args.pth_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint.pop("pro", None)

    model = source.UpCunet2x()
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    wrapper = CoreMLUpCunet2x(model).eval()
    example_input = torch.rand(1, 3, args.block_size, args.block_size)
    traced = torch.jit.trace(wrapper, example_input, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input", shape=example_input.shape, dtype=np.float32
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.pth_path))[0]
        output = f"{base}.mlpackage"
    mlmodel.save(output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
