"""Tile-by-tile comparison between the segmentation model and OSM. Independent of QGIS.

For every tile: run the model, rasterize the OSM features on the same pixel grid, and compute
the per-class IoU. Results are also assembled into two class mosaics (prediction and OSM).
"""
from dataclasses import dataclass, field

import numpy as np
from rasterio.transform import from_bounds

# Shared with the training code (tools/), see the plugin's __init__.py for the import path.
import auto_masking
import segmentation_metrics

from .onnx_segmenter import OnnxSegmenter

CRS = "EPSG:3857"
# Value of the mosaic pixels that were not evaluated (transparent or skipped tiles).
NODATA = 255


@dataclass
class TileResult:
    tile: object
    valid_fraction: float
    ious: dict
    mean_iou: float


@dataclass
class PipelineResult:
    """Mosaics are (H, W) uint8 class indices; bounds are (left, bottom, right, top) in CRS."""
    bounds: tuple
    predicted: np.ndarray
    osm: np.ndarray
    tile_results: list = field(default_factory=list)
    skipped_tiles: int = 0


class Canceled(Exception):
    pass


def run_pipeline(source, model_path, min_valid_fraction=0.5,
                 osm_features=None, query_path=None, highway_classes_path=None,
                 log=print, progress=lambda percent: None, is_canceled=lambda: False):
    """Compares the model prediction with OSM on every tile of a tile source.

    Args:
        source: XyzTileSource or GeoTiffTileSource (tiles in EPSG:3857, aligned on a grid).
        model_path: ONNX model exported by tools/export_onnx.py.
        min_valid_fraction: Tiles with a smaller fraction of non-transparent pixels are skipped.
        osm_features: Optional GeoDataFrame in EPSG:3857 (e.g. from QGIS vector layers).
            If None, the features are downloaded from Overpass in a single query.
        query_path: Overpass QL template used when osm_features is None.
        highway_classes_path: highway.txt listing the highway values drawn as roads.
        log: Callable receiving progress messages.
        progress: Callable receiving a percentage in [0, 100].
        is_canceled: Callable returning True when the user canceled the run.

    Returns:
        A PipelineResult, or None if no tile could be evaluated.

    Raises:
        Canceled: If is_canceled() became True during the run.
    """
    tiles = source.tiles
    log(source.description)

    # First pass, alpha only: find the tiles worth evaluating, and the extent for Overpass.
    kept = []
    for tile in tiles:
        alpha = tile.read_rgba()[..., 3]
        valid_fraction = np.count_nonzero(alpha) / alpha.size
        if valid_fraction >= min_valid_fraction:
            kept.append((tile, valid_fraction))
    skipped = len(tiles) - len(kept)
    log(f"{len(kept)} tiles kept, {skipped} skipped (less than {min_valid_fraction:.0%} valid pixels)")
    if not kept:
        return None
    progress(5)

    if osm_features is None:
        log("Downloading OSM features from Overpass...")
        osm_features = auto_masking.fetch_osm_features(_union_bounds(t for t, _ in kept), CRS, query_path)
    log(f"{len(osm_features)} OSM features")
    if highway_classes_path is None:
        highway_classes = auto_masking.load_highway_classes()
    else:
        highway_classes = auto_masking.load_highway_classes(highway_classes_path)
    progress(15)

    log("Loading model...")
    segmenter = OnnxSegmenter(model_path)

    # Mosaic grid: every tile has the same size and is aligned on the source's grid.
    tile_h, tile_w = kept[0][0].read_rgba().shape[:2]
    min_col = min(t.col for t, _ in kept)
    min_row = min(t.row for t, _ in kept)
    cols = max(t.col for t, _ in kept) - min_col + 1
    rows = max(t.row for t, _ in kept) - min_row + 1
    predicted_mosaic = np.full((rows * tile_h, cols * tile_w), NODATA, dtype=np.uint8)
    osm_mosaic = np.full_like(predicted_mosaic, NODATA)

    result = PipelineResult(_union_bounds(t for t, _ in kept), predicted_mosaic, osm_mosaic,
                            skipped_tiles=skipped)

    for i, (tile, valid_fraction) in enumerate(kept):
        if is_canceled():
            raise Canceled()

        rgba = tile.read_rgba()
        valid = rgba[..., 3] > 0
        predicted = segmenter.predict(rgba[..., :3])

        transform = from_bounds(*tile.bounds, tile_w, tile_h)
        osm_rgb = auto_masking.rasterize_osm_features(osm_features, tile.bounds, transform,
                                                      tile_h, tile_w, highway_classes)
        osm = segmentation_metrics.rgb_mask_to_classes(osm_rgb)

        ious, mean_iou = segmentation_metrics.per_class_iou(predicted, osm, valid)
        result.tile_results.append(TileResult(tile, valid_fraction, ious, mean_iou))

        row = (tile.row - min_row) * tile_h
        col = (tile.col - min_col) * tile_w
        predicted_mosaic[row:row + tile_h, col:col + tile_w] = np.where(valid, predicted, NODATA)
        osm_mosaic[row:row + tile_h, col:col + tile_w] = np.where(valid, osm, NODATA)

        progress(15 + 85 * (i + 1) / len(kept))

    mean = np.mean([r.mean_iou for r in result.tile_results])
    log(f"Done: mean IoU over {len(result.tile_results)} tiles = {mean:.3f}")
    return result


def _union_bounds(tiles):
    bounds = np.array([t.bounds for t in tiles])
    return (bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max())
