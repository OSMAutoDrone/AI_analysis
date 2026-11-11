# OSM Segmentation IoU (QGIS plugin)

Compares the U-Net segmentation (building, water, road) of an orthophoto with OpenStreetMap and
maps the IoU of each tile.

## Pipeline

```
Orthophoto GeoTIFF → reprojected to EPSG:3857 at the target resolution → square tiles
   (or Z/X/Y tile folder → zoom closest to the target resolution)
            → skip mostly transparent tiles
            → OSM features (one Overpass query, or loaded vector layers)
            → per tile: ONNX prediction + OSM rasterization → per-class IoU
            → layers: IoU heatmap, segmentation raster, OSM mask raster
```

## Inputs

- **Orthophoto (GeoTIFF)**, e.g. ODM's `odm_orthophoto.tif`: pick a raster layer of the project, or
  use `...` to load a file (it is added to the project as a layer). The image is reprojected on the
  fly to EPSG:3857 (like the training data) at the target resolution, then cut into a regular grid of
  *tile size* × *tile size* pixels starting at its top-left corner. The parts of edge tiles outside
  the image count as transparent.
- **Tile folder (Z/X/Y)**: `<z>/<x>/<y>.png` as written by OpenDroneMap; the zoom level closest to
  the target resolution is used, one tile = one heatmap cell.

The model was trained at about 0.5 m/px: keep the target resolution close to that.

## Outputs

| Layer | Content |
|---|---|
| IoU heatmap | One polygon per tile: `iou_mean`, `iou_building`, `iou_water`, `iou_road`, `valid_frac`. Red = low IoU, green = high IoU. |
| Segmentation | Model classes; background is transparent. |
| OSM mask | OSM classes rasterized like the training masks (`tools/auto_masking.py`). |

IoU rules (`tools/segmentation_metrics.py`):
- transparent pixels are ignored;
- a class absent from both the prediction and OSM gets a NULL IoU and does not count in the mean;
- a tile with none of the classes in either mask gets `iou_mean = 1`.

The rasters are temporary files; use the export options to keep them.

## Code layout

| Path | Role |
|---|---|
| `osm_segmentation_iou.py` | Menu entry, dialog wiring, adds the result layers |
| `osm_segmentation_iou_dialog_base.ui` | Dialog layout (edit with Qt Designer) |
| `segmentation_task.py` | Runs the pipeline in the background (`QgsTask`) |
| `output_layers.py` | GeoTIFF writing, layer creation and styles |
| `vector_input.py` | Converts loaded OSM vector layers to a GeoDataFrame |
| `core/` | QGIS-independent pipeline: tile sources (`geotiff_source.py`, `tile_source.py`), ONNX inference, IoU loop |
| `shared/` (deployed only) | Copies of `tools/auto_masking.py`, `overpass_router.py`, `segmentation_metrics.py` |

## Installation

1. **Export the model** (training venv, from the project root):
   ```
   python tools/export_onnx.py
   ```
2. **Install the missing packages in QGIS's Python**: open *OSGeo4W Shell* as administrator and run:
   ```
   python -m pip install onnxruntime rasterio overpy mercantile "numpy<2"
   ```
   `numpy<2` stops pip from upgrading numpy, which would break QGIS's GDAL bindings.
3. **Deploy the plugin** (from the project root):
   ```
   python qgis_plugin/deploy.py
   ```
   or `python qgis_plugin/deploy.py --zip` to build a ZIP for *Install from ZIP*.
4. Restart QGIS and enable *OSM Segmentation IoU* in *Plugins > Manage and Install Plugins*.

## Known limitations

- Tiles are processed one by one, but the two result mosaics are kept in memory
  (about 2 bytes per pixel at the target resolution).
- One Overpass query covers the whole area: very large orthophotos may hit the server timeout.
- Vector layers must have OSM tags as fields (`highway`, `building`, `natural`, `amenity`), as QuickOSM does.
