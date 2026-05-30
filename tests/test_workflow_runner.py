import unittest

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from shapely.ops import transform

import workflow_runner


class Wgs84BoundsMaskTests(unittest.TestCase):

    def test_condition_mask_treats_source_nodata_as_false(self):
        value = np.array(
            [
                [5, -9999],
                [0, 7],
            ],
            dtype=np.int16,
        )

        result = workflow_runner.condition_mask_op(value, -9999, "value > 0")

        np.testing.assert_array_equal(
            result,
            np.array(
                [
                    [1, 0],
                    [0, 1],
                ],
                dtype=np.uint8,
            ),
        )

    def test_downstream_coverage_mask_ignores_tiny_convolution_noise(self):
        mask = np.array(
            [
                [0.0, np.finfo(float).eps],
                [200 * np.finfo(float).eps, 1.0],
            ],
            dtype=np.float64,
        )

        result = workflow_runner.downstream_coverage_mask(mask)

        np.testing.assert_array_equal(
            result,
            np.array(
                [
                    [False, False],
                    [True, True],
                ]
            ),
        )

    def test_bounds_mask_includes_wgs84_pixel_centers(self):
        mask = workflow_runner._rasterize_wgs84_bounds_mask(
            (-1.5, -0.5, 1.5, 1.5),
            from_origin(-2, 2, 1, 1),
            "EPSG:4326",
            4,
            4,
        )
        expected = np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_bounds_mask_avoids_projected_full_extent_polygon_collapse(self):
        target_crs = CRS.from_proj4(
            "+proj=aeqd +lat_0=12.33225837 +lon_0=-0.0569542 "
            "+datum=WGS84 +units=m +no_defs"
        )
        full_extent_bounds = (-179.113, -54.822, 179.0, 79.487)
        full_extent_polygon = box(*full_extent_bounds)
        transformer = Transformer.from_crs(
            "EPSG:4326", target_crs, always_xy=True
        )
        projected_polygon = transform(transformer.transform, full_extent_polygon)

        india_lon_lat = (77.0, 22.0)
        india_x, india_y = transformer.transform(*india_lon_lat)

        self.assertTrue(
            full_extent_polygon.covers(Point(*india_lon_lat)),
            "India sample point should be inside the generated WGS84 AOI.",
        )
        self.assertFalse(
            projected_polygon.covers(Point(india_x, india_y)),
            "Projecting the near-global AOI rectangle collapses the useful "
            "rasterized area around the projection center.",
        )

        india_center_transform = from_origin(india_x - 0.5, india_y + 0.5, 1, 1)
        mask = workflow_runner._rasterize_wgs84_bounds_mask(
            full_extent_bounds,
            india_center_transform,
            target_crs,
            1,
            1,
        )
        self.assertEqual(mask[0, 0], 1)


if __name__ == "__main__":
    unittest.main()
