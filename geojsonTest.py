import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

base_path = r"C:/Users/ti-da/OneDrive/Documents/PMC/AI_analysis/data/"

with rasterio.open(f"{base_path}testGeoImage02.tif") as img:
    bounds = img.bounds
    raster_crs = img.crs
    data = img.read()

raster_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

data = np.moveaxis(data, 0, -1)

gdf = gpd.read_file(
    f"{base_path}brompton_highway.gpkg",
    layer="brompton_highway_lines"
)
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)
gdf = gdf[['osm_id', 'highway', 'name', 'geometry']]
gdf_in = gpd.clip(gdf, raster_bbox)

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(
    data,
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
