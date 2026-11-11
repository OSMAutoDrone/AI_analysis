"""Turns a PipelineResult into QGIS layers (rasters + IoU heatmap), with their styles and exports."""
from osgeo import gdal, osr
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsGraduatedSymbolRenderer,
    QgsPalettedRasterRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsRendererRange,
    QgsSymbol,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import Qt, QVariant
from qgis.PyQt.QtGui import QColor

import segmentation_metrics

from .core.pipeline import CRS, NODATA

HEATMAP_FIELD = "iou_mean"
HEATMAP_CLASSES = 10
HEATMAP_OPACITY = 140  # 0-255


def write_class_raster(array, bounds, path):
    """Writes a class-index mosaic as a georeferenced, LZW-compressed GeoTIFF."""
    height, width = array.shape
    left, bottom, right, top = bounds
    # Scoped exceptions: gdal.UseExceptions() would change GDAL's behavior for all of QGIS.
    with gdal.ExceptionMgr(), osr.ExceptionMgr():
        dataset = gdal.GetDriverByName("GTiff").Create(path, width, height, 1, gdal.GDT_Byte,
                                                       options=["COMPRESS=LZW"])
        dataset.SetGeoTransform((left, (right - left) / width, 0, top, 0, -(top - bottom) / height))
        srs = osr.SpatialReference()
        srs.SetFromUserInput(CRS)
        dataset.SetProjection(srs.ExportToWkt())
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(NODATA)
        band.WriteArray(array)
        dataset.FlushCache()
        dataset = None  # Closes the file.


def class_raster_layer(path, name):
    """Loads a class raster with one color per class; background is transparent."""
    layer = QgsRasterLayer(path, name)
    classes = []
    for idx, (class_name, rgb) in enumerate(segmentation_metrics.CLASSES):
        alpha = 0 if idx == 0 else 255
        classes.append(QgsPalettedRasterRenderer.Class(idx, QColor(*rgb, alpha), class_name))
    layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
    return layer


def heatmap_layer(result, name):
    """Builds an in-memory polygon layer with one cell per evaluated tile and its IoUs."""
    layer = QgsVectorLayer(f"Polygon?crs={CRS}", name, "memory")
    class_names = [n for i, (n, _) in enumerate(segmentation_metrics.CLASSES)
                   if i in segmentation_metrics.SCORED_CLASSES]
    fields = [QgsField("tile", QVariant.String), QgsField(HEATMAP_FIELD, QVariant.Double)]
    fields += [QgsField(f"iou_{n}", QVariant.Double) for n in class_names]
    fields += [QgsField("valid_frac", QVariant.Double)]
    provider = layer.dataProvider()
    provider.addAttributes(fields)
    layer.updateFields()

    features = []
    for tile_result in result.tile_results:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromRect(QgsRectangle(*tile_result.tile.bounds)))
        # None becomes NULL: the class is absent from both the prediction and OSM.
        feature.setAttributes([tile_result.tile.name, tile_result.mean_iou]
                              + [tile_result.ious[n] for n in class_names]
                              + [tile_result.valid_fraction])
        features.append(feature)
    provider.addFeatures(features)
    layer.updateExtents()

    apply_heatmap_style(layer)
    return layer


def apply_heatmap_style(layer):
    """Graduated style on iou_mean in [0, 1]: red = low IoU (disagreement), green = high."""
    ranges = []
    step = 1 / HEATMAP_CLASSES
    for i in range(HEATMAP_CLASSES):
        lower, upper = i * step, (i + 1) * step
        ratio = i / (HEATMAP_CLASSES - 1)
        if ratio < 0.5:  # Red -> yellow
            color = QColor(255, int(255 * ratio * 2), 0, HEATMAP_OPACITY)
        else:  # Yellow -> green
            color = QColor(int(255 * (1 - (ratio - 0.5) * 2)), 255, 0, HEATMAP_OPACITY)

        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(color)
        symbol.symbolLayer(0).setStrokeStyle(Qt.NoPen)
        ranges.append(QgsRendererRange(lower, upper, symbol, f"{lower:.1f} - {upper:.1f}"))

    layer.setRenderer(QgsGraduatedSymbolRenderer(HEATMAP_FIELD, ranges))
    layer.triggerRepaint()


def export_vector_layer(layer, path):
    """Saves a vector layer as GeoPackage; returns an error message or None."""
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    error, message, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, path, QgsProject.instance().transformContext(), options)
    return None if error == QgsVectorFileWriter.NoError else message
