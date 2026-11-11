"""Reads an orthophoto exported as a Z/X/Y tile folder (e.g. OpenDroneMap's tiles/).

Each tile is one processing unit ("dalle"). Folder layout: <root>/<z>/<x>/<y>.png, using the
XYZ (Google) row convention, which is what ODM writes. A TMS folder would need y flipped.

Tile sources share a small interface used by the pipeline: a `tiles` list (each tile has
name, col, row, bounds in EPSG:3857 and read_rgba()), a `description` and close().
See geotiff_source.py for the other source.
"""
import math
import os
from dataclasses import dataclass

import mercantile
import numpy as np
from PIL import Image

TILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# Equatorial ground resolution of zoom 0 for 256 px tiles, in m/px.
ZOOM0_RESOLUTION = 2 * math.pi * 6378137 / 256


@dataclass
class Tile:
    z: int
    x: int
    y: int
    path: str

    @property
    def name(self):
        return f"{self.z}/{self.x}/{self.y}"

    # Position in the tile grid, used to assemble the mosaics.
    @property
    def col(self):
        return self.x

    @property
    def row(self):
        return self.y

    @property
    def bounds(self):
        """(left, bottom, right, top) in EPSG:3857."""
        b = mercantile.xy_bounds(self.x, self.y, self.z)
        return b.left, b.bottom, b.right, b.top

    def read_rgba(self):
        """Returns the tile as (H, W, 4) uint8; tiles without alpha are fully opaque."""
        return np.array(Image.open(self.path).convert("RGBA"))


class XyzTileSource:
    """All tiles of the zoom level closest to target_resolution (m/px)."""

    def __init__(self, tile_dir, target_resolution):
        zoom = choose_zoom(tile_dir, target_resolution)
        self.tiles = list_tiles(tile_dir, zoom)
        self.description = f"Tile folder, zoom {zoom}: {len(self.tiles)} tiles"

    def close(self):
        pass


def find_zoom_levels(tile_dir):
    return sorted(int(d) for d in os.listdir(tile_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(tile_dir, d)))


def list_tiles(tile_dir, z):
    tiles = []
    zoom_dir = os.path.join(tile_dir, str(z))
    for x_name in os.listdir(zoom_dir):
        x_dir = os.path.join(zoom_dir, x_name)
        if not (x_name.isdigit() and os.path.isdir(x_dir)):
            continue
        for file_name in os.listdir(x_dir):
            stem, ext = os.path.splitext(file_name)
            if stem.isdigit() and ext.lower() in TILE_EXTENSIONS:
                tiles.append(Tile(z, int(x_name), int(stem), os.path.join(x_dir, file_name)))
    return sorted(tiles, key=lambda t: (t.y, t.x))


def ground_resolution(z, latitude):
    """Ground size of one pixel in m/px for 256 px tiles at this zoom and latitude."""
    return ZOOM0_RESOLUTION * math.cos(math.radians(latitude)) / 2 ** z


def choose_zoom(tile_dir, target_resolution):
    """Picks the zoom level whose ground resolution is closest to target_resolution (m/px).

    Raises:
        ValueError: If the folder contains no zoom level or no tile.
    """
    zooms = find_zoom_levels(tile_dir)
    if not zooms:
        raise ValueError(f"No zoom level folder found in {tile_dir}")

    def resolution(z):
        tiles = list_tiles(tile_dir, z)
        if not tiles:
            return math.inf
        latitude = mercantile.ul(tiles[0].x, tiles[0].y, z).lat
        return ground_resolution(z, latitude)

    return min(zooms, key=lambda z: abs(resolution(z) - target_resolution))
