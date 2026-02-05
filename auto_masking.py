import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.geometry import Point
from scipy.ndimage import binary_dilation

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
    road_mask = binary_dilation(
        road_mask,
        iterations=18
    ).astype(np.uint8)
    mask[road_mask == 1] = [240, 30, 44]
    return mask

def road_polygon_masking(gdf, mask):
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

if __name__ == "__main__":
    base_data_path = r"./data/"
    highway_data_map_path = "OSM-class-lists/highway.txt"

    with rasterio.open(f"{base_data_path}testGeoImage.tif") as img:
        bounds = img.bounds
        raster_crs = img.crs
        transform = img.transform
        height = img.height
        width = img.width
        data = img.read()

    raster_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    data = np.moveaxis(data, 0, -1)

    gdf = gpd.read_file(
        f"{base_data_path}brompton_highway_parking.gpkg",
        layer="brompton_highway_parking_lines"
    )

    gdf_poly = gpd.read_file(
        f"{base_data_path}brompton_highway_parking.gpkg",
        layer="brompton_highway_parking_multipolygons"
    )

    with open(highway_data_map_path, 'r') as f:
        highway_class_list = f.read().splitlines()

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    if gdf_poly.crs != raster_crs:
        gdf_poly = gdf_poly.to_crs(raster_crs)
    gdf = gdf[['osm_id', 'highway', 'name', 'geometry']]
    gdf_poly = gdf_poly[['osm_id','amenity', 'highway', 'geometry']]
    gdf = gdf[gdf['highway'].isin(highway_class_list)]
    gdf_poly = gdf_poly[(gdf_poly['highway'].isin(highway_class_list)) | (gdf_poly['amenity'] == 'parking')]
    gdf_in = gpd.clip(gdf, raster_bbox)
    gdf_poly_in = gdf_poly.clip(gdf_poly, raster_bbox)


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
    mask = road_polygon_masking(gdf_poly_in, mask)
    gdf_points = gpd.GeoDataFrame(geometry=points, crs=raster_crs)


    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
    )

    ax.imshow(
        mask,
        alpha=0.5,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top)
        )

    gdf_in.plot(
        ax=ax,
        color="red",
        linewidth=1
    )

    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    plt.show()



