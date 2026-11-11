"""QGIS plugin comparing a segmentation model's output with OpenStreetMap, tile by tile."""
import os
import sys


def _add_shared_modules_to_path():
    """Makes the modules shared with the training code (auto_masking, ...) importable.

    deploy.py copies them into <plugin>/shared. When the plugin runs straight from the
    repository (e.g. through a symlink), they are read from AI_analysis/tools instead.
    They use flat imports (e.g. "from overpass_router import ..."), hence sys.path.
    """
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(plugin_dir, "shared"),
                      os.path.join(plugin_dir, "..", "..", "tools")):
        if os.path.isdir(candidate):
            path = os.path.normpath(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


_add_shared_modules_to_path()


def classFactory(iface):  # pylint: disable=invalid-name
    from .osm_segmentation_iou import OSMSegmentationIoU
    return OSMSegmentationIoU(iface)
