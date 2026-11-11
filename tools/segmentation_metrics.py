"""Class definitions and per-class IoU shared by IoU_inference.py and the QGIS plugin.

Numpy only, so it runs both in the training venv and in QGIS's Python.
"""
import numpy as np

# Index -> (name, RGB). Same order and colors as training.data_setup.MaskColorMap
# and auto_masking, so class indices match the model output channels.
CLASSES = [
    ("unlabelled", (0, 0, 0)),
    ("building", (255, 255, 0)),
    ("water", (0, 64, 255)),
    ("road", (240, 30, 44)),
]
# Class 0 is the background: it is not scored.
SCORED_CLASSES = range(1, len(CLASSES))


def rgb_mask_to_classes(mask_rgb):
    """Converts an (H, W, 3) RGB mask to (H, W) class indices; unknown colors become 0."""
    classes = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    for idx, (_, color) in enumerate(CLASSES):
        classes[np.all(mask_rgb == np.array(color), axis=-1)] = idx
    return classes


def per_class_iou(predicted, reference, valid=None):
    """Computes the IoU of each scored class between two class-index masks.

    Args:
        predicted: (H, W) class indices from the model.
        reference: (H, W) class indices from OSM.
        valid: Optional (H, W) bool mask; pixels set to False (e.g. transparent
            orthophoto pixels) are ignored.

    Returns:
        (ious, mean_iou). ious maps class name -> IoU, or None when the class is absent
        from both masks (its IoU is undefined). mean_iou averages the defined IoUs and is
        1.0 when no scored class appears in either mask (agreement on "nothing").
    """
    if valid is not None:
        predicted = predicted[valid]
        reference = reference[valid]

    ious = {}
    for idx in SCORED_CLASSES:
        pred_c = predicted == idx
        ref_c = reference == idx
        union = np.count_nonzero(pred_c | ref_c)
        name = CLASSES[idx][0]
        ious[name] = None if union == 0 else float(np.count_nonzero(pred_c & ref_c) / union)

    defined = [v for v in ious.values() if v is not None]
    mean_iou = float(np.mean(defined)) if defined else 1.0
    return ious, mean_iou
