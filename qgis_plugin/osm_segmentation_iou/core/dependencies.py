"""Checks the Python packages the plugin needs but QGIS does not ship."""
import importlib.util

# Import name -> pip package name.
REQUIRED_PACKAGES = {
    "onnxruntime": "onnxruntime",
    "rasterio": "rasterio",
    "overpy": "overpy",
    "mercantile": "mercantile",
    "geopandas": "geopandas",
    "scipy": "scipy",
    "PIL": "pillow",
}
# QGIS's GDAL bindings are built against numpy 1.x: pip must not upgrade it.
NUMPY_PIN = "numpy<2"


def missing_packages():
    return [pip_name for module, pip_name in REQUIRED_PACKAGES.items()
            if importlib.util.find_spec(module) is None]


def install_command(packages):
    return "python -m pip install " + " ".join(packages) + f' "{NUMPY_PIN}"'
