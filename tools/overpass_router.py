import overpy
import requests
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid
import time
from pathlib import Path

# Resolved from this file so the module works regardless of the working directory.
DEFAULT_QUERY_PATH = Path(__file__).resolve().parent.parent / "data" / "OSM_request.txt"

# Tried in order; the next one is used when a server fails.
# overpass-api.de rejected automated requests (HTTP 406/502) in October 2026.
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
MAX_ATTEMPTS = 5
FIRST_RETRY_DELAY_S = 2
# Client-side limit per request: without it an overloaded server can hang for minutes.
# Slightly above the [timeout:40] of OSM_request.txt so the server can answer first.
REQUEST_TIMEOUT_S = 60
# Public Overpass servers reject anonymous automated clients.
USER_AGENT = "Perdoudes-AI-analysis/0.1 (OSM segmentation IoU research)"


class OverpassQueryRejected(Exception):
    pass


def overpass_query(crs, bounds, query_path=DEFAULT_QUERY_PATH, urls=OVERPASS_URLS):
    """Downloads the OSM features inside bounds as a GeoDataFrame (EPSG:4326).

    Each attempt cycles to the next server; the delay between attempts doubles.

    Args:
        crs: CRS of bounds.
        bounds: (left, bottom, right, top) in crs.
        query_path: Overpass QL template, where "BOUNDS" is replaced by the bbox.
        urls: Overpass interpreter endpoints, tried in order.

    Raises:
        OverpassQueryRejected: If the server rejects the query itself (HTTP 400).
        RuntimeError: If every attempt failed.
    """
    with open(query_path) as query_f:
        query = query_f.read()
    bbox = convert_crs(crs, bounds)
    query = query.replace("BOUNDS", f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")

    delay = FIRST_RETRY_DELAY_S
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        url = urls[attempt % len(urls)]
        try:
            # requests instead of overpy.query: overpy has no timeout nor custom headers.
            response = requests.post(url, data={"data": query}, timeout=REQUEST_TIMEOUT_S,
                                     headers={"User-Agent": USER_AGENT})
            if response.status_code == 400:
                # Retrying cannot fix a malformed query.
                raise OverpassQueryRejected(f"Overpass rejected the query: {response.text[:500]}")
            response.raise_for_status()
            result = overpy.Overpass().parse_json(response.content)
            print(f"Successful query ({url})")
            return overpy_to_gdf(result)
        except OverpassQueryRejected:
            raise
        except Exception as e:
            last_error = e
            print(f"--- Overpass error on {url}: {type(e).__name__}: {e}. Retry in {delay} s ---")
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Overpass query failed after {MAX_ATTEMPTS} attempts: {last_error}")


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