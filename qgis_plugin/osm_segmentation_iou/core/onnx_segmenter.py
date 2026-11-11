"""Runs the U-Net exported by tools/export_onnx.py with onnxruntime (no PyTorch needed)."""
import numpy as np
import onnxruntime as ort

# The U-Net downsamples 5 times: input sizes must be multiples of 2**5.
SIZE_MULTIPLE = 32


class OnnxSegmenter:
    """Predicts a class index per pixel from an RGB uint8 image."""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, rgb):
        """Returns (H, W) uint8 class indices for an (H, W, 3) uint8 RGB image."""
        h, w = rgb.shape[:2]
        pad_h = -h % SIZE_MULTIPLE
        pad_w = -w % SIZE_MULTIPLE
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        # Same scaling as training.utils.np_to_tensor: [0, 255] -> [0, 1], no mean/std.
        x = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        probabilities = self.session.run(None, {self.input_name: x})[0]
        return probabilities[0].argmax(axis=0).astype(np.uint8)[:h, :w]
