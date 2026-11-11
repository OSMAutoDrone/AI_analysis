"""Cuts a georeferenced orthophoto (e.g. ODM's odm_orthophoto.tif) into tiles for the pipeline.

The orthophoto is reprojected on the fly to EPSG:3857 at the target ground resolution, like
the training data, then split into a regular grid of square tiles anchored at its top-left
corner. Edge tiles extend past the image: that part is transparent (alpha = 0), so the
pipeline's valid-pixel threshold decides whether they are kept.
"""
import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

CRS = "EPSG:3857"


@dataclass
class GeoTiffTile:
    source: "GeoTiffTileSource"
    col: int
    row: int

    @property
    def name(self):
        return f"r{self.row}_c{self.col}"

    @property
    def bounds(self):
        """(left, bottom, right, top) in EPSG:3857."""
        size = self.source.tile_size * self.source.pixel_size
        left = self.source.left + self.col * size
        top = self.source.top - self.row * size
        return left, top - size, left + size, top

    def read_rgba(self):
        return self.source.read_rgba(self)


class GeoTiffTileSource:
    """Exposes a GeoTIFF as a grid of tile_size x tile_size tiles in EPSG:3857.

    Args:
        path: Orthophoto path (RGB or RGBA; nodata or alpha marks the empty area).
        target_resolution: Ground resolution of the tiles in m/px.
        tile_size: Tile width and height in pixels (multiple of 32 for the U-Net).
    """

    def __init__(self, path, target_resolution, tile_size=256):
        self.tile_size = tile_size
        self.dataset = rasterio.open(path)

        left, bottom, right, top = transform_bounds(self.dataset.crs, CRS, *self.dataset.bounds)
        # Web Mercator stretches distances by 1 / cos(latitude): a ground resolution of
        # target_resolution needs target_resolution / cos(lat) map units per pixel.
        _, lat_bottom, _, lat_top = transform_bounds(self.dataset.crs, "EPSG:4326", *self.dataset.bounds)
        latitude = (lat_bottom + lat_top) / 2
        self.pixel_size = target_resolution / math.cos(math.radians(latitude))

        self.left, self.top = left, top
        self.cols = math.ceil((right - left) / (self.pixel_size * tile_size))
        self.rows = math.ceil((top - bottom) / (self.pixel_size * tile_size))

        has_alpha = ColorInterp.alpha in self.dataset.colorinterp
        # The VRT covers the whole tile grid, so every tile window is inside it.
        # Average resampling: we downsample a lot (ODM is ~3 cm/px) and want no aliasing.
        self.vrt = WarpedVRT(
            self.dataset, crs=CRS,
            transform=from_origin(left, top, self.pixel_size, self.pixel_size),
            width=self.cols * tile_size, height=self.rows * tile_size,
            resampling=Resampling.average, add_alpha=not has_alpha,
        )
        self.tiles = [GeoTiffTile(self, col, row) for row in range(self.rows) for col in range(self.cols)]
        self.description = (f"Orthophoto reprojected to {CRS} at {target_resolution} m/px: "
                            f"{self.cols} x {self.rows} tiles of {tile_size} px")

    def read_rgba(self, tile):
        window = Window(tile.col * self.tile_size, tile.row * self.tile_size, self.tile_size, self.tile_size)
        # The VRT has 4 bands: RGB + alpha (the source's own or the one added from nodata).
        data = self.vrt.read(indexes=[1, 2, 3, self.vrt.count], window=window)
        return np.moveaxis(data, 0, -1)

    def close(self):
        self.vrt.close()
        self.dataset.close()
