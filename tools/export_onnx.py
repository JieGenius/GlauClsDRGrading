from mmpretrain.registry import MODELS
from mmengine import Config
import torch


def main():
    cfg = Config.fromfile("../configs/convnext/convnext-tiny_b32_refuge_aptos.py")
    model = MODELS.build(cfg.model)
    model.load_state_dict(torch.load("../work_dirs/convnext-tiny_b32_refuge_aptos/epoch_48.pth")["state_dict"])
    input = torch.randn(1, 3, 448, 448)
    print(model(input))
    with torch.no_grad():
        torch.onnx.export(model,  # model being run
                          input,  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          verbose=False,
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=15,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=["output"],  # the model's output names
                          dynamic_axes=None  # variable length axes
                          )

if __name__ == "__main__":
    main()