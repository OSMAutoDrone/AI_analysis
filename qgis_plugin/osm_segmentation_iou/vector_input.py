"""Converts OSM vector layers loaded in QGIS (QuickOSM, .osm, .gpkg...) to the GeoDataFrame
format expected by auto_masking.rasterize_osm_features.

Must run in the main thread: QGIS layers are not thread-safe.
"""
import geopandas as gpd
from shapely import wkb
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

# Only these OSM tags are used by auto_masking; a layer without one of them just has it empty.
TAG_FIELDS = ["name", "highway", "amenity", "building", "natural"]


def layers_to_geodataframe(layers, crs="EPSG:3857"):
    """Merges the line and polygon features of several layers, reprojected to crs.

    Points are ignored. 'type' is set to 'linestring' or 'polygon' from the geometry.
    """
    target_crs = QgsCoordinateReferenceSystem(crs)
    rows, geometries = [], []
    for layer in layers:
        transform = QgsCoordinateTransform(layer.crs(), target_crs, QgsProject.instance())
        field_names = set(layer.fields().names())
        for feature in layer.getFeatures():
            geometry = QgsGeometry(feature.geometry())
            geometry_type = QgsWkbTypes.geometryType(geometry.wkbType())
            if geometry_type == QgsWkbTypes.LineGeometry:
                osm_type = "linestring"
            elif geometry_type == QgsWkbTypes.PolygonGeometry:
                osm_type = "polygon"
            else:
                continue

            geometry.transform(transform)
            geometries.append(wkb.loads(bytes(geometry.asWkb())))
            row = {tag: _to_python(feature[tag]) if tag in field_names else None for tag in TAG_FIELDS}
            row["id"] = feature.id()
            row["type"] = osm_type
            rows.append(row)

    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)


def _to_python(value):
    # Depending on the QGIS version, NULL attributes come back as None or as a null QVariant.
    if value is None or (isinstance(value, QVariant) and value.isNull()):
        return None
    return value
