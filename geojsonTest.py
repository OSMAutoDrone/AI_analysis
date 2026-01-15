import time
import geopandas as gpd

file_path = r"C:/Users/ti-da/OneDrive/Documents/PMC/Comparison/data/brompton_all.gpkg"
current = time.time()

gdf = gpd.read_file(file_path, layer='brompton_all_multilinestrings')
columns = gdf.columns.to_list()
print(columns)
pass