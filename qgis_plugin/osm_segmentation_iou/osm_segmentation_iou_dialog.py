import os

from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from qgis.PyQt import uic
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QListWidgetItem

# The layout lives in the .ui file: edit it with Qt Designer.
FORM_CLASS, _ = uic.loadUiType(
    os.path.join(os.path.dirname(__file__), "osm_segmentation_iou_dialog_base.ui")
)


class OSMSegmentationIoUDialog(QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.browseTileFolderButton.clicked.connect(
            lambda: self._browse_folder(self.tileFolderInput, "Select the Z/X/Y tile folder"))
        self.browseFolderButton.clicked.connect(
            lambda: self._browse_folder(self.outputFolderInput, "Select the export folder"))
        self.browseModelButton.clicked.connect(self._browse_model)
        self.browseOrthophotoButton.clicked.connect(self._browse_orthophoto)

    def log(self, text):
        self.logOutput.append(text)

    def refresh_vector_layers(self):
        """Lists the project's vector layers as checkable items (kept checked if still present)."""
        checked = set(layer.id() for layer in self.checked_vector_layers())
        self.vectorLayersList.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            item = QListWidgetItem(layer.name())
            item.setData(Qt.UserRole, layer.id())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer.id() in checked else Qt.Unchecked)
            self.vectorLayersList.addItem(item)

    def refresh_raster_layers(self, select_id=None):
        """Lists the project's file-based raster layers; keeps (or sets) the selection."""
        if select_id is None:
            select_id = self.rasterLayerCombo.currentData()
        self.rasterLayerCombo.clear()
        for layer in QgsProject.instance().mapLayers().values():
            # Only GDAL rasters have a file the pipeline can read (not WMS/XYZ basemaps).
            if isinstance(layer, QgsRasterLayer) and layer.providerType() == "gdal":
                self.rasterLayerCombo.addItem(layer.name(), layer.id())
        index = self.rasterLayerCombo.findData(select_id)
        if index >= 0:
            self.rasterLayerCombo.setCurrentIndex(index)

    def selected_raster_layer(self):
        layer_id = self.rasterLayerCombo.currentData()
        return QgsProject.instance().mapLayer(layer_id) if layer_id else None

    def checked_vector_layers(self):
        project = QgsProject.instance()
        layers = []
        for i in range(self.vectorLayersList.count()):
            item = self.vectorLayersList.item(i)
            layer = project.mapLayer(item.data(Qt.UserRole))
            if item.checkState() == Qt.Checked and layer is not None:
                layers.append(layer)
        return layers

    def set_running(self, running):
        self.runButton.setEnabled(not running)
        self.cancelButton.setEnabled(running)

    def _browse_folder(self, line_edit, title):
        folder = QFileDialog.getExistingDirectory(self, title, line_edit.text())
        if folder:
            line_edit.setText(folder)

    def _browse_orthophoto(self):
        """Loads an orthophoto file as a project layer, so it is displayed and selectable."""
        path, _ = QFileDialog.getOpenFileName(self, "Select the orthophoto", "",
                                              "GeoTIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return
        layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
        if not layer.isValid():
            self.log(f"Could not open {path} as a raster.")
            return
        QgsProject.instance().addMapLayer(layer)
        self.refresh_raster_layers(select_id=layer.id())

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select the ONNX model", self.modelPathInput.text(),
                                              "ONNX model (*.onnx)")
        if path:
            self.modelPathInput.setText(path)
