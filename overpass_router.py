import overpy
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid
import time
base_data_path = r"./data/"


def overpass_query(crs, bounds):
    api = overpy.Overpass()
    with open(f"{base_data_path}OSM_request.txt") as query_f:
        query = query_f.read()
    bbox = convert_crs(crs, bounds)
    query = query.replace("BOUNDS", f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")

    succes = False
    while not succes:
        try:
            result = api.query(query)
            succes = True
        except Exception as e:
            print("--- SERVER ERROR WHILE QUERYING, RETRY IN 5 SECONDS ---")
            time.sleep(5)

    print("Succesfull query")
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

    geometries = []
    attributes = []
    way_index = {way.id: way for way in result.ways}
    for way in result.ways:
        coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
        if coords[0] == coords[-1] and len(coords) > 2:
            # Closed way, likely a Polygon
            geometry = Polygon(coords)
            type = 'polygon'
        elif len(coords) > 1:
            # LineString
            geometry = LineString(coords)
            type = 'linestring'
        else:
            continue

        geometries.append(geometry)
        # Store relevant attributes (tags, id, etc.)
        attributes.append({
            'id': way.id,
            'name': way.tags.get('name', None),
            'highway': way.tags.get('highway', None),
            'amenity': way.tags.get('amenity', None),
            'building': way.tags.get('building', None),
            'natural': way.tags.get('natural', None),
            'type': type
        })
    for relation in result.relations:
        outer_lines = []

        for member in relation.members:
            if not isinstance(member, overpy.RelationWay):
                continue
            if member.role != "outer":
                continue

            way = way_index.get(member.ref)
            if way is None or not way.nodes:
                continue

            coords = [(float(n.lon), float(n.lat)) for n in way.nodes]
            if len(coords) >= 2:
                outer_lines.append(LineString(coords))

        if not outer_lines:
            continue

        try:
            # unary_union soude les segments, polygonize referme les anneaux
            merged = unary_union(outer_lines)
            polys = list(polygonize(merged))

            if not polys:
                continue

            geometry = make_valid(unary_union(polys))

            geometries.append(geometry)
            attributes.append({
                "id": relation.id,
                "name": relation.tags.get("name"),
                "highway": relation.tags.get("highway"),
                "amenity": relation.tags.get("amenity"),
                "building": relation.tags.get("building"),
                "natural": relation.tags.get("natural"),
                "type": "polygon",
            })

        except Exception as e:
            print(f"  ✗ Relation {relation.id} ignorée : {e}")
    # 3. Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs="EPSG:4326")


    return gdf