import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.geometry import Point
from scipy.ndimage import binary_dilation
from overpass_router import overpass_query
import os

def road_masking(gdf, mask):
    if gdf.empty:
        return mask
    road_mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    m_per_pixel = transform.a*unit_per_pixel
    road_mask = binary_dilation(
        road_mask,
        iterations=int(3.5/m_per_pixel)
    ).astype(np.uint8)
    mask[road_mask == 1] = [240, 30, 44]
    return mask

def road_polygon_masking(gdf, mask):
    gdf = gdf[(gdf['highway'].isin(highway_class_list)) | (gdf['amenity'] == 'parking')]
    gdf = gdf_poly.clip(gdf, raster_bbox)
    if gdf.empty:
        return mask
    road_mask = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    mask[road_mask == 1] = [240, 30, 44]
    return mask

def building_polygon_masking(gdf, mask):
    gdf_building = gdf.dropna(subset=['building'])
    gdf_building = gdf_building.clip(gdf_building, raster_bbox)
    if gdf.empty:
        return mask
    building_mask = rasterize(
        [(geom, 1) for geom in gdf_building.geometry if geom is not None],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    mask[building_mask == 1] = [255, 255, 0]
    return mask

def water_polygon_masking(gdf, mask):
    gdf_water = gdf[gdf['natural'] == 'water']
    gdf_water = gdf_water.clip(gdf, raster_bbox)
    if gdf.empty:
        return mask
    water_mask = rasterize(
        [(geom, 1) for geom in gdf_water.geometry if geom is not None],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    mask[water_mask == 1] = [0, 64, 255]
    return mask

if __name__ == "__main__":
    img_name = "testGeoImage02.tif"
    base_data_path = r"./data/"
    highway_data_map_path = "OSM-class-lists/highway.txt"

    unit_per_pixel = 0.5

    with rasterio.open(os.path.join(base_data_path,img_name)) as img:
        bounds = img.bounds
        raster_crs = img.crs
        transform = img.transform
        height = img.height
        width = img.width
        data = img.read()
        if abs(transform.a) != abs(transform.e):
            print("WARNING- Image pixel's transformation does not match")

    raster_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    data = np.moveaxis(data, 0, -1)


    gdf, gdf_poly = overpass_query(raster_crs, bounds)

    with open(highway_data_map_path, 'r') as f:
        highway_class_list = f.read().splitlines()

    if gdf.crs != raster_crs or gdf.crs != raster_crs :
        print("WARNING- raster and img crs don't match")
        gdf = gdf.to_crs(raster_crs)
        gdf_poly = gdf_poly.to_crs(raster_crs)
    gdf = gdf[['osm_id', 'highway', 'name', 'geometry']]
    gdf_poly = gdf_poly[['osm_id','amenity','natural', 'highway', 'building', 'geometry']]
    gdf = gdf[gdf['highway'].isin(highway_class_list)]
    gdf_in = gpd.clip(gdf, raster_bbox)
    gdf_poly_in = gpd.clip(gdf_poly, raster_bbox)
    # Extraire tous les points (sommets) des géométries
    points = []

    for geom in gdf_in.geometry:
        if geom.geom_type == "LineString":
            points.extend([Point(xy) for xy in geom.coords])

        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                points.extend([Point(xy) for xy in line.coords])

    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask = road_masking(gdf_in, mask)
    mask = road_polygon_masking(gdf_poly, mask)
    mask = building_polygon_masking(gdf_poly, mask)
    mask = water_polygon_masking(gdf_poly, mask)
    gdf_points = gpd.GeoDataFrame(geometry=points, crs=raster_crs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    # --- Image seule ---
    axes[0].imshow(
        data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
    )
    axes[0].set_title("Image")
    axes[0].axis("off")

    # --- Masque seul ---
    axes[1].imshow(
        mask,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
    )
    axes[1].set_title("Mask")
    axes[1].axis("off")

    # --- Overlay ---
    axes[2].imshow(
        data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
    )

    axes[2].imshow(
        mask,
        alpha=0.5,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
    )

    gdf_in.plot(
        ax=axes[2],
        color="red",
        linewidth=1
    )

    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(base_data_path,img_name.replace(".tif", ".png")))
    plt.show()




