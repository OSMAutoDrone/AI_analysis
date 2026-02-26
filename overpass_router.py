import overpy
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid
base_data_path = r"./data/"


def overpass_query(crs, bounds):
    api = overpy.Overpass()
    with open(f"{base_data_path}OSM_request.txt") as query_f:
        query = query_f.read()
    bbox = convert_crs(crs, bounds)
    query = query.replace("BOUNDS", f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
    print(query)
    result = api.query(query)
    return overpy_to_gdf(result)


def convert_crs(source_crs, bounds):
    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True
    )

    west, south = transformer.transform(bounds[0], bounds[1])
    east, north = transformer.transform(bounds[2], bounds[3])

    return (south, west, north, east)

def overpy_to_gdf(result):

    ways = []
    polygons = []

    def extract_tags(osm_obj):

        tags = osm_obj.tags

        return {
            "osm_id": osm_obj.id,
            "highway": tags.get("highway"),
            "amenity": tags.get("amenity"),
            "natural": tags.get("natural"),
            "building": tags.get("building"),
            "landuse": tags.get("landuse"),
            "name": tags.get("name"),
        }

    # -------- Ways --------

    for way in result.ways:

        coords = [(float(n.lon), float(n.lat)) for n in way.nodes]

        if len(coords) < 2:
            continue

        data = extract_tags(way)

        if coords[0] == coords[-1] and len(coords) >= 4:

            data["geometry"] = Polygon(coords)
            polygons.append(data)

        else:

            data["geometry"] = LineString(coords)
            ways.append(data)


    # -------- Relations (CRITIQUE) --------

    for rel in result.relations:

        data = extract_tags(rel)

        outer = []

        for member in rel.members:

            if member.role == "outer":

                way = member.resolve()

                coords = [(float(n.lon), float(n.lat)) for n in way.nodes]

                if len(coords) >= 4:

                    outer.append(coords)

        if outer:

            data["geometry"] = Polygon(outer[0])

            polygons.append(data)


    gdf_ways = gpd.GeoDataFrame(ways, crs="EPSG:4326")

    gdf_polygons = gpd.GeoDataFrame(polygons, crs="EPSG:4326")


    return gdf_ways, gdf_polygons