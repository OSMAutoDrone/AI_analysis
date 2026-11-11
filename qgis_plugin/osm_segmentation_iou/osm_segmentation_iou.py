"""Plugin entry point: menu action, dialog wiring, and loading the results into the project."""
import os
import shutil
import tempfile
from datetime import datetime

from qgis.core import QgsApplication, QgsProject, QgsSettings
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox

from .core.dependencies import install_command, missing_packages
from .osm_segmentation_iou_dialog import OSMSegmentationIoUDialog

PLUGIN_DIR = os.path.dirname(__file__)
MENU = "&OSM Segmentation IoU"
SETTINGS_PREFIX = "osm_segmentation_iou/"
DEFAULT_MODEL_PATH = os.path.join(PLUGIN_DIR, "models", "model.onnx")
# Copied here by deploy.py. When absent (running from the repository), the defaults
# of the shared tools/ modules are used instead.
QUERY_PATH = os.path.join(PLUGIN_DIR, "resources", "OSM_request.txt")
HIGHWAY_CLASSES_PATH = os.path.join(PLUGIN_DIR, "resources", "highway.txt")


class OSMSegmentationIoU:

    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.task = None

    def initGui(self):
        icon = QIcon(os.path.join(PLUGIN_DIR, "icon.png"))
        self.action = QAction(icon, "OSM Segmentation IoU", self.iface.mainWindow())
        self.action.triggered.connect(self.open_dialog)
        self.iface.addPluginToMenu(MENU, self.action)

    def unload(self):
        self.iface.removePluginMenu(MENU, self.action)

    def open_dialog(self):
        missing = missing_packages()
        if missing:
            QMessageBox.warning(
                self.iface.mainWindow(), "OSM Segmentation IoU",
                "Missing Python packages: " + ", ".join(missing) + "\n\n"
                "Install them from the OSGeo4W Shell (run as administrator), then restart QGIS:\n\n"
                + install_command(missing))
            return

        if self.dialog is None:
            self.dialog = OSMSegmentationIoUDialog()
            self.dialog.runButton.clicked.connect(self.run_analysis)
            self.dialog.cancelButton.clicked.connect(self.cancel_analysis)
            self._load_settings()

        self.dialog.refresh_raster_layers()
        self.dialog.refresh_vector_layers()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def run_analysis(self):
        # Imported here: these modules need the packages checked in open_dialog.
        from .segmentation_task import SegmentationTask

        dialog = self.dialog
        dialog.logOutput.clear()
        dialog.progressBar.setValue(0)

        source_factory = self._source_factory()
        if source_factory is None:
            return
        model_path = dialog.modelPathInput.text().strip()
        if not os.path.isfile(model_path):
            dialog.log("Please select an existing ONNX model.")
            return

        pipeline_kwargs = {
            "model_path": model_path,
            "min_valid_fraction": dialog.minValidSpin.value() / 100,
            "query_path": QUERY_PATH if os.path.isfile(QUERY_PATH) else None,
            "highway_classes_path": HIGHWAY_CLASSES_PATH if os.path.isfile(HIGHWAY_CLASSES_PATH) else None,
        }

        if dialog.vectorLayersRadio.isChecked():
            from .vector_input import layers_to_geodataframe

            layers = dialog.checked_vector_layers()
            if not layers:
                dialog.log("Please check at least one vector layer.")
                return
            dialog.log("Reading OSM vector layers...")
            pipeline_kwargs["osm_features"] = layers_to_geodataframe(layers)

        self._save_settings()
        # Temporary rasters; QGIS layers point to them, so they are kept until the OS cleans them.
        output_dir = tempfile.mkdtemp(prefix="osm_segmentation_iou_")

        self.task = SegmentationTask(source_factory, pipeline_kwargs, output_dir, self.on_task_finished)
        self.task.log_message.connect(dialog.log)
        self.task.progressChanged.connect(lambda p: dialog.progressBar.setValue(int(p)))
        dialog.set_running(True)
        QgsApplication.taskManager().addTask(self.task)

    def _source_factory(self):
        """Returns a callable creating the tile source chosen in the dialog, or None if invalid.

        The source itself is created later, in the task's worker thread.
        """
        from .core.geotiff_source import GeoTiffTileSource
        from .core.tile_source import XyzTileSource

        dialog = self.dialog
        resolution = dialog.targetResolutionSpin.value()
        if dialog.geotiffRadio.isChecked():
            layer = dialog.selected_raster_layer()
            if layer is None:
                dialog.log("Please select an orthophoto layer (or load one with ...).")
                return None
            # For the gdal provider, the source is the file path (plus options after '|').
            path = layer.source().split("|")[0]
            tile_size = dialog.tileSizeSpin.value()
            return lambda: GeoTiffTileSource(path, resolution, tile_size)

        tile_dir = dialog.tileFolderInput.text().strip()
        if not os.path.isdir(tile_dir):
            dialog.log("Please select an existing tile folder.")
            return None
        return lambda: XyzTileSource(tile_dir, resolution)

    def cancel_analysis(self):
        if self.task is not None:
            self.task.cancel()

    def on_task_finished(self, task):
        from .output_layers import class_raster_layer, export_vector_layer, heatmap_layer

        dialog = self.dialog
        dialog.set_running(False)
        self.task = None
        if task.error is not None:
            dialog.log(f"Failed: {task.error}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project = QgsProject.instance()
        group = project.layerTreeRoot().insertGroup(0, f"OSM Segmentation IoU ({timestamp})")

        layers = [heatmap_layer(task.result, "IoU heatmap")]
        if dialog.addPredictionCheckbox.isChecked():
            layers.append(class_raster_layer(task.prediction_path, "Segmentation"))
        if dialog.addOsmCheckbox.isChecked():
            layers.append(class_raster_layer(task.osm_path, "OSM mask"))
        for layer in layers:
            # addMapLayer(..., False) then addLayer: places the layer in the group only.
            project.addMapLayer(layer, False)
            group.addLayer(layer)
        dialog.log("Layers added to the project.")

        self._export(task, layers[0], timestamp, export_vector_layer)
        dialog.log("Done!")

    def _export(self, task, heatmap, timestamp, export_vector_layer):
        dialog = self.dialog
        export_rasters = dialog.exportRastersCheckbox.isChecked()
        export_heatmap = dialog.exportHeatmapCheckbox.isChecked()
        if not (export_rasters or export_heatmap):
            return

        output_dir = dialog.outputFolderInput.text().strip()
        if not output_dir:
            dialog.log("Export skipped: no export folder selected.")
            return
        os.makedirs(output_dir, exist_ok=True)

        if export_rasters:
            for source, name in ((task.prediction_path, "segmentation"), (task.osm_path, "osm_mask")):
                destination = os.path.join(output_dir, f"{name}_{timestamp}.tif")
                shutil.copyfile(source, destination)
                dialog.log(f"Saved: {destination}")

        if export_heatmap:
            destination = os.path.join(output_dir, f"iou_heatmap_{timestamp}.gpkg")
            error = export_vector_layer(heatmap, destination)
            dialog.log(f"Heatmap export failed: {error}" if error else f"Saved: {destination}")

    def _load_settings(self):
        settings = QgsSettings()
        dialog = self.dialog
        dialog.tileFolderInput.setText(settings.value(SETTINGS_PREFIX + "tile_folder", ""))
        dialog.modelPathInput.setText(settings.value(SETTINGS_PREFIX + "model_path", DEFAULT_MODEL_PATH))
        dialog.outputFolderInput.setText(settings.value(SETTINGS_PREFIX + "output_folder", ""))
        dialog.tileSizeSpin.setValue(int(settings.value(SETTINGS_PREFIX + "tile_size", 256)))
        if settings.value(SETTINGS_PREFIX + "input_mode", "geotiff") == "xyz":
            dialog.xyzRadio.setChecked(True)

    def _save_settings(self):
        settings = QgsSettings()
        dialog = self.dialog
        settings.setValue(SETTINGS_PREFIX + "tile_folder", dialog.tileFolderInput.text())
        settings.setValue(SETTINGS_PREFIX + "model_path", dialog.modelPathInput.text())
        settings.setValue(SETTINGS_PREFIX + "output_folder", dialog.outputFolderInput.text())
        settings.setValue(SETTINGS_PREFIX + "tile_size", dialog.tileSizeSpin.value())
        settings.setValue(SETTINGS_PREFIX + "input_mode", "geotiff" if dialog.geotiffRadio.isChecked() else "xyz")
