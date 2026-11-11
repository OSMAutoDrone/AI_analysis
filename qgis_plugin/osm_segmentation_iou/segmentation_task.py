"""Background task running the pipeline, so QGIS stays responsive during inference."""
import os
import traceback

from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from .core.pipeline import Canceled, run_pipeline
from .output_layers import write_class_raster


class SegmentationTask(QgsTask):
    """Runs run_pipeline and writes the two class rasters in a worker thread.

    Layers are not created here: QGIS layers must be created in the main thread, which
    on_finished does (finished() is called in the main thread by QGIS).
    """

    # Signals are queued across threads, so the dialog can safely display them.
    log_message = pyqtSignal(str)

    def __init__(self, source_factory, pipeline_kwargs, output_dir, on_finished):
        """
        Args:
            source_factory: Callable returning the tile source. Called in the worker thread,
                since opening and reprojecting a large orthophoto can take time.
            pipeline_kwargs: Keyword arguments for run_pipeline (without source and callbacks).
            output_dir: Folder receiving prediction.tif and osm.tif.
            on_finished: Called as on_finished(task) once the task ends, in the main thread.
        """
        super().__init__("OSM Segmentation IoU", QgsTask.CanCancel)
        self.source_factory = source_factory
        self.pipeline_kwargs = pipeline_kwargs
        self.output_dir = output_dir
        self.on_finished = on_finished
        self.result = None
        self.error = None
        self.prediction_path = os.path.join(output_dir, "prediction.tif")
        self.osm_path = os.path.join(output_dir, "osm.tif")

    def run(self):
        source = None
        try:
            source = self.source_factory()
            self.result = run_pipeline(source, **self.pipeline_kwargs, log=self.log_message.emit,
                                       progress=self.setProgress, is_canceled=self.isCanceled)
            if self.result is None:
                self.error = "No tile to evaluate (all tiles are below the valid pixel threshold)."
                return False

            self.log_message.emit("Writing rasters...")
            write_class_raster(self.result.predicted, self.result.bounds, self.prediction_path)
            write_class_raster(self.result.osm, self.result.bounds, self.osm_path)
            return True
        except Canceled:
            self.error = "Canceled."
            return False
        except Exception:
            self.error = traceback.format_exc()
            return False
        finally:
            if source is not None:
                source.close()

    def finished(self, succeeded):
        self.on_finished(self)
