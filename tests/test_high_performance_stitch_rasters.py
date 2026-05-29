import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.high_performance_stitch_rasters import (
    read_raster_list,
    stitch_rasters,
)


class HighPerformanceStitchRastersTests(unittest.TestCase):

    def _write_raster(self, path, array, left, top, nodata=-1):
        profile = {
            "driver": "GTiff",
            "height": array.shape[0],
            "width": array.shape[1],
            "count": 1,
            "dtype": array.dtype,
            "crs": "EPSG:4326",
            "transform": from_origin(left, top, 1, 1),
            "nodata": nodata,
        }
        with rasterio.open(path, "w", **profile) as target:
            target.write(array, 1)

    def test_stitches_adjacent_rasters_and_skips_nodata(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            raster_a_path = workspace_path / "a.tif"
            raster_b_path = workspace_path / "b.tif"
            output_path = workspace_path / "stitched.tif"

            self._write_raster(
                raster_a_path,
                np.array([[1, -1], [3, 4]], dtype=np.int16),
                0,
                2,
            )
            self._write_raster(
                raster_b_path,
                np.array([[5, 6], [-1, 8]], dtype=np.int16),
                2,
                2,
            )

            stitch_rasters([raster_a_path, raster_b_path], output_path)

            with rasterio.open(output_path) as stitched:
                result = stitched.read(1)
                self.assertEqual(stitched.nodata, -1)
                self.assertEqual(stitched.width, 4)
                self.assertEqual(stitched.height, 2)

            np.testing.assert_array_equal(
                result,
                np.array([[1, -1, 5, 6], [3, 4, -1, 8]], dtype=np.int16),
            )

    def test_later_rasters_overwrite_earlier_valid_pixels(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            raster_a_path = workspace_path / "a.tif"
            raster_b_path = workspace_path / "b.tif"
            output_path = workspace_path / "stitched.tif"

            self._write_raster(
                raster_a_path,
                np.array([[1, 2], [3, 4]], dtype=np.int16),
                0,
                2,
            )
            self._write_raster(
                raster_b_path,
                np.array([[9, -1], [-1, 12]], dtype=np.int16),
                0,
                2,
            )

            stitch_rasters([raster_a_path, raster_b_path], output_path)

            with rasterio.open(output_path) as stitched:
                result = stitched.read(1)

            np.testing.assert_array_equal(
                result,
                np.array([[9, 2], [3, 12]], dtype=np.int16),
            )

    def test_reads_list_relative_to_list_file(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            raster_path = workspace_path / "a.tif"
            list_path = workspace_path / "rasters.txt"
            self._write_raster(
                raster_path,
                np.array([[1]], dtype=np.int16),
                0,
                1,
            )
            list_path.write_text(
                "\n# comment\n" + raster_path.name + "\n",
                encoding="utf-8",
            )

            self.assertEqual(read_raster_list(list_path), [raster_path])


if __name__ == "__main__":
    unittest.main()
