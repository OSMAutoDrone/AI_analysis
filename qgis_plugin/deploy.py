"""Assembles the plugin (code + shared modules + resources + model) and installs or zips it.

The plugin folder in the repository does not contain the modules shared with the training
code (tools/) nor the OSM resource files: this script copies them in, so the deployed plugin
is self-contained.

Usage, from the project root:
    python qgis_plugin/deploy.py            # install into the default QGIS profile
    python qgis_plugin/deploy.py --zip      # build dist/osm_segmentation_iou.zip instead
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_NAME = "osm_segmentation_iou"
PLUGIN_SRC = PROJECT_ROOT / "qgis_plugin" / PLUGIN_NAME

SHARED_MODULES = ["auto_masking.py", "overpass_router.py", "segmentation_metrics.py"]
RESOURCES = [PROJECT_ROOT / "data" / "OSM_request.txt", PROJECT_ROOT / "OSM-class-lists" / "highway.txt"]
DEFAULT_PLUGINS_DIR = Path(os.environ.get("APPDATA", "")) / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"


def assemble(destination):
    """Copies the full plugin into destination/<PLUGIN_NAME>, replacing any previous copy."""
    target = Path(destination) / PLUGIN_NAME
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(PLUGIN_SRC, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    (target / "shared").mkdir()
    for module in SHARED_MODULES:
        shutil.copy2(PROJECT_ROOT / "tools" / module, target / "shared" / module)

    (target / "resources").mkdir(exist_ok=True)
    for resource in RESOURCES:
        shutil.copy2(resource, target / "resources" / resource.name)

    if not (target / "models" / "model.onnx").exists():
        print("WARNING: models/model.onnx is missing, run tools/export_onnx.py first.")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install or package the OSM Segmentation IoU plugin.")
    parser.add_argument("--plugins-dir", default=str(DEFAULT_PLUGINS_DIR),
                        help="QGIS plugins folder (default: the 'default' QGIS profile)")
    parser.add_argument("--zip", action="store_true", help="Build dist/osm_segmentation_iou.zip instead")
    args = parser.parse_args()

    if args.zip:
        with tempfile.TemporaryDirectory() as tmp:
            assemble(tmp)
            archive = shutil.make_archive(str(PROJECT_ROOT / "dist" / PLUGIN_NAME), "zip", tmp, PLUGIN_NAME)
        print(f"Built {archive} (install it with Plugins > Manage and Install Plugins > Install from ZIP)")
    else:
        print(f"Installed in {assemble(args.plugins_dir)}. Restart QGIS or reload the plugin.")
