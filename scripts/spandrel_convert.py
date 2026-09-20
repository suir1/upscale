import torch
import coremltools as ct
import numpy as np
import argparse
import os
from spandrel import MAIN_REGISTRY, ModelLoader


class FixedSizeModel(torch.nn.Module):
    def __init__(self, model, size):
        super().__init__()
        self.model = model
        self.size = size

    def forward(self, x):
        factor = getattr(self.model, "shuffle_factor", None)
        if factor:
            scale = self.model.scale // factor
            x = torch.pixel_unshuffle(x, factor)
            x = self.model.model(x)
            return x[:, :, :self.size * scale, :self.size * scale]
        return self.model(x)

def main():
    parser = argparse.ArgumentParser(description="Convert a .pth model to .mlpackage with spandrel")
    parser.add_argument("pth_path", help="Path to the .pth model file")
    args = parser.parse_args()

    model = ModelLoader().load_from_file(args.pth_path).model
    model.eval()

    size = 256
    example_input = torch.randn(1, 3, size, size)

    traced = torch.jit.trace(FixedSizeModel(model, size).eval(), example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input", shape=example_input.shape, dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="output", dtype=np.float32)
        ],
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS16
    )

    base = os.path.basename(args.pth_path)
    name, _ = os.path.splitext(base)
    mlmodel_name = f"{name}.mlpackage"
    mlmodel.save(mlmodel_name)
    print(f"Saved: {mlmodel_name}")

if __name__ == "__main__":
    main()
