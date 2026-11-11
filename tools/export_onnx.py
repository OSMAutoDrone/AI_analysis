"""Exports the trained U-Net (.pth) to ONNX for the QGIS plugin, which runs it with onnxruntime.

Run from the project root with the training venv:
    python tools/export_onnx.py --checkpoint training/UNet_Model_Output/model_full_v02.pth \
        --output qgis_plugin/osm_segmentation_iou/models/model.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

# Must match the architecture used for training (training/train.py, IoU_inference.load_model).
ENCODER = 'efficientnet-b4'
ACTIVATION = 'softmax2d'
NUM_CLASSES = 4


def build_model(checkpoint_path):
    # encoder_weights=None: the checkpoint overwrites them, no need to download ImageNet weights.
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=NUM_CLASSES, activation=ACTIVATION)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def export(checkpoint_path, output_path, opset=17):
    """Exports the model with dynamic height/width and checks it against PyTorch.

    The input is (N, 3, H, W) float32 RGB in [0, 1] (same scaling as training.utils.np_to_tensor),
    with H and W multiples of 32. The output is (N, 4, H, W) class probabilities.
    """
    model = build_model(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.rand(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["image"], output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                      "probabilities": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
        dynamo=False,
    )
    print(f"Exported: {output_path}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: skipped the PyTorch/ONNX comparison.")
        return

    # Check on a different size than the export one to validate the dynamic axes.
    check = torch.rand(1, 3, 480, 640)
    with torch.no_grad():
        expected = model(check).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": check.numpy()})[0]
    same_classes = np.mean(expected.argmax(1) == actual.argmax(1))
    print(f"Max abs difference: {np.abs(expected - actual).max():.2e}, same argmax: {same_classes:.4%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="training/UNet_Model_Output/model_full_v02.pth")
    parser.add_argument("--output", default="qgis_plugin/osm_segmentation_iou/models/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.opset)
