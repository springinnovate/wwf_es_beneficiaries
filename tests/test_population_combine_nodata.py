import unittest

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

import workflow_runner


class PopulationCombineNodataTests(unittest.TestCase):

    def test_vrt_fill_nodata_does_not_conflict_with_zero_population(self):
        profile = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": from_origin(0, 2, 1, 1),
            "nodata": None,
        }
        source_array = np.array([[0, 5], [0, 0]], dtype=np.float32)

        with MemoryFile() as memfile:
            with memfile.open(**profile) as source:
                source.write(source_array, 1)

            with memfile.open() as source:
                with WarpedVRT(
                    source,
                    crs=source.crs,
                    transform=source.transform,
                    width=4,
                    height=2,
                    resampling=Resampling.nearest,
                    nodata=workflow_runner.POPULATION_COMBINE_VRT_NODATA,
                ) as source_vrt:
                    incoming = source_vrt.read(1).astype(np.float32, copy=False)

        self.assertEqual(incoming[0, 0], 0)
        self.assertEqual(incoming[0, 1], 5)
        self.assertEqual(incoming[0, 2], workflow_runner.POPULATION_COMBINE_VRT_NODATA)

        incoming[incoming < 0] = 0
        np.testing.assert_array_equal(
            incoming,
            np.array([[0, 5, 0, 0], [0, 0, 0, 0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
