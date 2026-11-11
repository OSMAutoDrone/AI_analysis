import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import box
from scipy.ndimage import binary_dilation
from overpass_router import overpass_query
import os
from pathlib import Path

# This module is shared with the QGIS plugin, whose Python has no cv2, albumentations,
# tqdm or matplotlib: those are imported only inside the functions that need them.

base_data_path = r"./data/"
DEFAULT_HIGHWAY_CLASSES_PATH = Path(__file__).resolve().parent.parent / "OSM-class-lists" / "highway.txt"
OSM_COLUMNS = ['id', 'name', 'highway', 'amenity', 'building', 'natural', 'type']

def road_masking(input_gdf, input_mask, highway_class_list, h, w, t):
    output_mask = input_mask.copy()
    gdf_road = input_gdf[(input_gdf['type'] == 'linestring') & (input_gdf['highway'].isin(highway_class_list))]
    gdf_road = gdf_road.dropna(subset=['highway'])
    if len(gdf_road) < 1:
        return output_mask
    road_mask = rasterize(
        [(geom, 1) for geom in gdf_road.geometry],
        out_shape=(h, w),
        transform=t,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    unit_per_pixel = 0.5
    m_per_pixel = t.a*unit_per_pixel
    road_mask = binary_dilation(
        road_mask,
        iterations=int(3.5/m_per_pixel)
    ).astype(np.uint8)
    output_mask[road_mask == 1] = [240, 30, 44]
    return output_mask

def road_polygon_masking(input_gdf, input_mask, h, w, t):
    output_mask = input_mask.copy()
    gdf_parking = input_gdf[input_gdf['type'] == 'polygon']
    gdf_parking = gdf_parking[(gdf_parking['highway'].notna()) | (gdf_parking['amenity'] == 'parking')]
    if len(gdf_parking) < 1:
        return output_mask
    road_mask = rasterize(
        [(geom, 1) for geom in gdf_parking.geometry if geom is not None],
        out_shape=(h, w),
        transform=t,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    output_mask[road_mask == 1] = [240, 30, 44]
    return output_mask

def building_polygon_masking(input_gdf, input_mask, h, w, t):
    output_mask = input_mask.copy()
    gdf_building = input_gdf.dropna(subset=['building'])
    if len(gdf_building) < 1:
        return output_mask
    building_mask = rasterize(
        [(geom, 1) for geom in gdf_building.geometry if geom is not None],
        out_shape=(h, w),
        transform=t,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    output_mask[building_mask == 1] = [255, 255, 0]
    return output_mask

def water_polygon_masking(input_gdf, input_mask, h, w, t):
    output_mask = input_mask.copy()
    gdf_water = input_gdf[input_gdf['natural'] == 'water']
    if len(gdf_water) < 1:
        return output_mask
    water_mask = rasterize(
        [(geom, 1) for geom in gdf_water.geometry if geom is not None],
        out_shape=(h, w),
        transform=t,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    output_mask[water_mask == 1] = [0, 64, 255]
    return output_mask

def debug_plot(input_data, input_mask, input_bounds, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        input_data,
        extent=(input_bounds.left, input_bounds.right, input_bounds.bottom, input_bounds.top)
    )

    ax.imshow(
        input_mask,
        alpha=0.5,
        extent=(input_bounds.left, input_bounds.right, input_bounds.bottom, input_bounds.top)
    )

    ax.set_title(title)
    ax.axis("off")

def masking_pipeline(img_name, target_size=None):
    """Builds the OSM mask of a georeferenced image.

    Args:
        img_name: Path to the georeferenced raster.
        target_size: Optional (width, height). If set, the image and the mask are
            resized like in preprocessing.py (bilinear for the image, nearest for the mask).
    """
    with rasterio.open(img_name) as img:
        bounds = img.bounds
        raster_crs = img.crs
        transform = img.transform
        height = img.height
        width = img.width
        data = img.read()
        if abs(transform.a) != abs(transform.e):
            print("WARNING- Image pixel's transformation does not match")

    data = np.moveaxis(data, 0, -1)

    mask = osm_masking(bounds, raster_crs, transform, height, width)

    if target_size is not None:
        from preprocessing import resize_pair
        data, mask = resize_pair(data, mask, target_size)

    return mask, data, bounds

def osm_masking(bounds, raster_crs, transform, height, width):
    """Downloads the OSM features inside bounds and rasterizes them into an RGB class mask.

    Args:
        bounds: (left, bottom, right, top) in raster_crs.
        raster_crs: CRS of bounds and transform.
        transform: Affine pixel-to-CRS transform of the target grid.
        height: Mask height in pixels.
        width: Mask width in pixels.
    """
    gdf = fetch_osm_features(bounds, raster_crs)
    return rasterize_osm_features(gdf, bounds, transform, height, width)

def fetch_osm_features(bounds, raster_crs, query_path=None):
    """Downloads the OSM features inside bounds from Overpass, reprojected to raster_crs.

    Split from rasterize_osm_features so a large area can be queried once and then
    rasterized tile by tile.

    Args:
        bounds: (left, bottom, right, top) in raster_crs.
        raster_crs: CRS of bounds and of the returned GeoDataFrame.
        query_path: Optional Overpass QL template; overpass_router's default otherwise.
    """
    if query_path is None:
        gdf = overpass_query(raster_crs, bounds)
    else:
        gdf = overpass_query(raster_crs, bounds, query_path=query_path)

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    return gdf

def load_highway_classes(path=DEFAULT_HIGHWAY_CLASSES_PATH):
    with open(path, 'r') as f:
        return f.read().splitlines()

def rasterize_osm_features(gdf, bounds, transform, height, width, highway_class_list=None):
    """Rasterizes OSM features into an RGB class mask (colors of MaskColorMap).

    Args:
        gdf: Features in the CRS of bounds, with the OSM_COLUMNS attributes
            ('type' is 'linestring' or 'polygon'). Missing columns are treated as empty.
        bounds: (left, bottom, right, top) of the mask.
        transform: Affine pixel-to-CRS transform of the mask.
        height: Mask height in pixels.
        width: Mask width in pixels.
        highway_class_list: highway values drawn as roads; read from highway.txt if None.

    Returns:
        (height, width, 3) uint8 RGB mask.
    """
    if highway_class_list is None:
        highway_class_list = load_highway_classes()

    raster_bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    gdf_in = gpd.clip(gdf, raster_bbox)
    for col in OSM_COLUMNS:
        if col not in gdf_in.columns:
            gdf_in[col] = None

    # Drawing order matters: later classes overwrite earlier ones (buildings on top).
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask = water_polygon_masking(gdf_in, mask, height, width, transform)
    mask = road_masking(gdf_in, mask, highway_class_list, height, width, transform)
    mask = road_polygon_masking(gdf_in, mask, height, width, transform)
    mask = building_polygon_masking(gdf_in, mask, height, width, transform)

    return mask

def build_dataset():
    """Builds the training images, masks and visualizations from data/source."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from tqdm import tqdm

    for d, p, f in os.walk(os.path.join(base_data_path, 'source')):
        for img in tqdm(f):
            print("Processing: ", img)
            img_path = os.path.join(base_data_path, 'source', img)
            mask, data, bounds = masking_pipeline(img_path)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

            # Image
            axes[0].imshow(
                data,
                extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
            )
            axes[0].set_title("Image")
            axes[0].axis("off")

            # Mask
            axes[1].imshow(
                mask,
                extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
            )
            axes[1].set_title("Mask")
            axes[1].axis("off")

            # Overlay
            axes[2].imshow(
                data,
                extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
            )

            axes[2].imshow(
                mask,
                alpha=0.3,
                extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
            )

            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(base_data_path, 'base/viz', img.replace(".tif", ".png")))
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255)
                data = data.astype(np.uint8)

            Image.fromarray(data).save(
                os.path.join(base_data_path, "base/img", img.replace(".tif", ".jpg")),
                quality=95,
                subsampling=0
            )

            mask_uint8 = mask.astype(np.uint8)

            Image.fromarray(mask_uint8).save(
                os.path.join(base_data_path, "base/mask", img.replace(".tif", ".png"))
            )

            plt.close()


if __name__ == "__main__":
    build_dataset()
