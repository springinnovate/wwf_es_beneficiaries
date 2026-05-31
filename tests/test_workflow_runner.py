import unittest
from pathlib import Path
import tempfile

import numpy as np
import rasterio
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

        condition_mask_op = workflow_runner.make_condition_mask_op(
            -9999,
            "value > 0",
        )
        result = condition_mask_op(value)

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

    def test_downstream_coverage_threshold_ignores_tiny_convolution_noise(self):
        mask = np.array(
            [
                [0.0, np.finfo(float).eps],
                [200 * np.finfo(float).eps, 1.0],
            ],
            dtype=np.float64,
        )

        result = mask > workflow_runner.DOWNSTREAM_COVERAGE_EPSILON

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

    def test_stitch_coverage_masks_uses_union_semantics(self):
        profile = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": from_origin(0, 2, 1, 1),
            "nodata": 0,
        }
        workflow_runner._set_tiled_geotiff_creation_options(profile)

        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            coverage_a_path = workspace_path / "coverage_a.tif"
            coverage_b_path = workspace_path / "coverage_b.tif"
            target_path = workspace_path / "stitched.tif"
            with rasterio.open(coverage_a_path, "w", **profile) as coverage_a:
                coverage_a.write(
                    np.array([[1, 0], [0, 0]], dtype=np.uint8),
                    1,
                )
            with rasterio.open(coverage_b_path, "w", **profile) as coverage_b:
                coverage_b.write(
                    np.array([[0, 1], [0, 0]], dtype=np.uint8),
                    1,
                )

            workflow_runner.stitch_coverage_masks(
                [
                    ("a", coverage_a_path),
                    ("b", coverage_b_path),
                ],
                1,
                workspace_path / "work",
                target_path,
            )

            with rasterio.open(target_path) as stitched:
                result = stitched.read(1)

        np.testing.assert_array_equal(
            result,
            np.array([[1, 1], [0, 0]], dtype=np.uint8),
        )

    def test_mask_population_with_coverage_applies_coverage_once(self):
        transform = from_origin(0, 2, 1, 1)
        coverage_profile = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": 0,
        }
        population_profile = {
            **coverage_profile,
            "dtype": "int32",
        }
        workflow_runner._set_tiled_geotiff_creation_options(coverage_profile)
        workflow_runner._set_tiled_geotiff_creation_options(population_profile)

        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            coverage_path = workspace_path / "coverage.tif"
            population_path = workspace_path / "population.tif"
            target_path = workspace_path / "masked_population.tif"
            with rasterio.open(coverage_path, "w", **coverage_profile) as coverage:
                coverage.write(
                    np.array([[1, 0], [1, 0]], dtype=np.uint8),
                    1,
                )
            with rasterio.open(population_path, "w", **population_profile) as pop:
                pop.write(
                    np.array([[10, 20], [30, 40]], dtype=np.int32),
                    1,
                )

            population_sum = workflow_runner.mask_population_with_coverage(
                population_path,
                coverage_path,
                target_path,
            )

            with rasterio.open(target_path) as masked_population:
                result = masked_population.read(1)

        np.testing.assert_array_equal(
            result,
            np.array([[10, 0], [30, 0]], dtype=np.int32),
        )
        self.assertEqual(population_sum, 40)


if __name__ == "__main__":
    unittest.main()
