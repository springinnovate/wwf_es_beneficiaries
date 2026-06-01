import unittest
import warnings
from unittest import mock
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import Point, box
from shapely.ops import transform

import workflow_runner


class Wgs84BoundsMaskTests(unittest.TestCase):

    def test_conditional_workflows_default_to_two_workers(self):
        config = {
            "inputs": {"taskgraph_workers": None},
            "masks": [
                {"type": "travel_time_population"},
                {"type": "conditional_raster"},
            ],
        }

        with mock.patch("workflow_runner.psutil.cpu_count", return_value=16):
            worker_count = workflow_runner.calculate_taskgraph_worker_count(
                config,
                work_unit_count=5421,
            )

        self.assertEqual(worker_count, 2)

    def test_configured_taskgraph_workers_override_default_and_cap_at_cpu_count(self):
        config = {
            "inputs": {"taskgraph_workers": 99},
            "masks": [{"type": "conditional_raster"}],
        }

        with mock.patch("workflow_runner.psutil.cpu_count", return_value=16):
            worker_count = workflow_runner.calculate_taskgraph_worker_count(
                config,
                work_unit_count=5421,
            )

        self.assertEqual(worker_count, 16)

    def test_is_eckert_iv_crs_detects_projection_without_proj4_warning(self):
        eckert_crs = CRS.from_proj4("+proj=eck4 +R=6371000 +units=m +no_defs")

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            self.assertTrue(workflow_runner._is_eckert_iv_crs(eckert_crs))

        self.assertFalse(workflow_runner._is_eckert_iv_crs("EPSG:4326"))
        warning_messages = [str(warning.message) for warning in captured_warnings]
        self.assertFalse(
            any("lose important projection information" in msg for msg in warning_messages)
        )

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

    def test_mask_raster_to_vector_sets_default_nodata_outside_geometry(self):
        raster_profile = {
            "driver": "GTiff",
            "height": 4,
            "width": 4,
            "count": 1,
            "dtype": "int16",
            "crs": "EPSG:4326",
            "transform": from_origin(0, 4, 1, 1),
        }
        workflow_runner._set_tiled_geotiff_creation_options(raster_profile)

        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            raster_path = workspace_path / "dem.tif"
            vector_path = workspace_path / "mask.gpkg"
            raster_array = np.arange(16, dtype=np.int16).reshape((4, 4))
            raster_array[1, 1] = 0
            with rasterio.open(raster_path, "w", **raster_profile) as raster:
                raster.write(raster_array, 1)
            gpd.GeoDataFrame(
                geometry=[box(0, 0, 2, 4)],
                crs="EPSG:4326",
            ).to_file(vector_path, driver="GPKG")

            workflow_runner.mask_raster_to_vector(raster_path, vector_path)

            with rasterio.open(raster_path) as masked_raster:
                masked_array = masked_raster.read(1)
                nodata = masked_raster.nodata

        self.assertEqual(nodata, np.iinfo(np.int16).min)
        self.assertEqual(masked_array[1, 1], 0)
        np.testing.assert_array_equal(
            masked_array[:, 2:],
            np.full((4, 2), np.iinfo(np.int16).min, dtype=np.int16),
        )

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

    def test_antimeridian_bounds_split_into_valid_wgs84_windows(self):
        bounds = (167.7408, 66.7660, -179.1999, 70.0103)

        split_bounds = workflow_runner._split_wgs84_antimeridian_bounds(bounds)

        self.assertEqual(
            split_bounds,
            [
                (167.7408, 66.7660, 180.0, 70.0103),
                (-180.0, 66.7660, -179.1999, 70.0103),
            ],
        )
        transform = from_origin(-180, 90, 1, 1)
        for bounds_part in split_bounds:
            window = workflow_runner._integer_window(
                from_bounds(*bounds_part, transform=transform),
                360,
                180,
            )
            self.assertGreater(window.width, 0)
            self.assertGreater(window.height, 0)
        pixel_size = 0.008333333333333
        self.assertAlmostEqual(
            workflow_runner._floor_to_grid(-180.0, pixel_size),
            -180.0,
        )
        self.assertAlmostEqual(
            workflow_runner._ceil_to_grid(180.0, pixel_size),
            180.0,
        )

    def test_windowed_travel_reach_matches_whole_raster_reach(self):
        profile = {
            "driver": "GTiff",
            "height": 6,
            "width": 6,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:3857",
            "transform": from_origin(0, 6, 1, 1),
            "nodata": 0,
        }
        mask_profile = {**profile, "dtype": "uint8"}
        workflow_runner._set_tiled_geotiff_creation_options(profile)
        workflow_runner._set_tiled_geotiff_creation_options(mask_profile)

        friction = np.ones((6, 6), dtype=np.float32)
        source_mask = np.zeros((6, 6), dtype=np.int8)
        source_mask[1, 1] = 1
        source_mask[4, 4] = 1
        expected = workflow_runner.shortest_distances.find_mask_reach(
            friction,
            source_mask,
            1.0,
            6,
            6,
            1.5,
            progress_interval_seconds=0,
        )

        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            friction_path = workspace_path / "friction.tif"
            mask_path = workspace_path / "mask.tif"
            coverage_path = workspace_path / "coverage.tif"
            with rasterio.open(friction_path, "w", **profile) as friction_raster:
                friction_raster.write(friction, 1)
            with rasterio.open(mask_path, "w", **mask_profile) as mask_raster:
                mask_raster.write(source_mask.astype(np.uint8), 1)

            workflow_runner.calculate_windowed_travel_reach(
                friction_path,
                mask_path,
                coverage_path,
                max_time_mins=1.5,
                buffer_pixels=2,
                core_block_size=2,
            )

            with rasterio.open(coverage_path) as coverage_raster:
                result = coverage_raster.read(1)

        np.testing.assert_array_equal(result, expected)

    def test_travel_reach_priority_queue_guard_reports_window_context(self):
        friction = np.ones((3, 3), dtype=np.float32)
        source_mask = np.zeros((3, 3), dtype=np.int8)
        source_mask[1, 1] = 1

        with self.assertRaisesRegex(
            MemoryError,
            "travel reach priority queue exceeded guard limit",
        ):
            workflow_runner.shortest_distances.find_mask_reach(
                friction,
                source_mask,
                1.0,
                3,
                3,
                2.0,
                progress_interval_seconds=0,
                queue_guard_multiplier=0,
            )

    def test_travel_reach_ignores_nan_friction(self):
        friction = np.ones((3, 3), dtype=np.float32)
        friction[1, 2] = np.nan
        source_mask = np.zeros((3, 3), dtype=np.int8)
        source_mask[1, 1] = 1

        result = workflow_runner.shortest_distances.find_mask_reach(
            friction,
            source_mask,
            1.0,
            3,
            3,
            2.0,
            progress_interval_seconds=0,
        )

        self.assertEqual(result[1, 2], 0)
        self.assertEqual(result[1, 1], 1)

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

    def test_full_extent_includes_travel_time_condition_rasters(self):
        raster_paths = workflow_runner._raster_paths_for_full_extent(
            {
                "inputs": {
                    "population_raster_path": Path("population.tif"),
                    "traveltime_raster_path": Path("travel.tif"),
                    "dem_raster_path": Path("dem.tif"),
                },
                "masks": [
                    {
                        "type": "travel_time_population",
                        "params": {
                            "condition_raster_path": "travel_sources.tif",
                        },
                    },
                    {
                        "type": "conditional_raster",
                        "params": {
                            "condition_raster_path": "downstream_sources.tif",
                        },
                    },
                ],
            }
        )

        self.assertIn(Path("travel_sources.tif"), raster_paths)
        self.assertIn(Path("downstream_sources.tif"), raster_paths)

    def test_travel_time_population_requires_condition_source_params(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            for filename in [
                "population.tif",
                "travel.tif",
                "dem.tif",
                "subwatersheds.gpkg",
            ]:
                (workspace_path / filename).touch()
            config_path = workspace_path / "missing_travel_source.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run_name: missing_travel_source",
                        "work_dir: work",
                        "output_dir: output",
                        "inputs:",
                        "  population_raster_path: "
                        f"{workspace_path / 'population.tif'}",
                        "  traveltime_raster_path: "
                        f"{workspace_path / 'travel.tif'}",
                        f"  dem_raster_path: {workspace_path / 'dem.tif'}",
                        "  subwatershed_vector_path: "
                        f"{workspace_path / 'subwatersheds.gpkg'}",
                        "  analyze_full_raster_extent: true",
                        "  wgs84_pixel_size: 1",
                        "  travel_time_pixel_size_m: 1000",
                        "  buffer_size_m: 5000",
                        "sections:",
                        "  - masks:",
                        "    - id: within_travel_time",
                        "      type: travel_time_population",
                        "      params:",
                        "        max_hours: 1",
                        "  - combine:",
                        "    - logic: OR",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "condition_raster_path, expression",
            ):
                workflow_runner.process_config(config_path)

    def test_travel_time_source_mask_uses_condition_raster_and_drain_mask(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            condition_path = workspace_path / "hotspots.tif"
            reference_path = workspace_path / "friction_clip.tif"
            vector_path = workspace_path / "drain.gpkg"
            target_path = workspace_path / "source_mask.tif"

            reference_info = {
                "pixel_size": (900, -900),
                "bounding_box": [0, 0, 1800, 1800],
                "projection_wkt": "PROJCS[...]",
            }
            condition_info = {
                "datatype": workflow_runner.gdal.GDT_Float32,
                "nodata": [-9999],
            }

            with mock.patch(
                "workflow_runner.geoprocessing.get_raster_info",
                side_effect=[reference_info, condition_info],
            ), mock.patch(
                "workflow_runner.geoprocessing.warp_raster"
            ) as warp_raster, mock.patch(
                "workflow_runner.geoprocessing.raster_calculator"
            ) as raster_calculator:
                result_path = workflow_runner.create_travel_time_source_mask(
                    condition_path,
                    "value > 0",
                    vector_path,
                    reference_path,
                    workspace_path,
                    target_path,
                )

            self.assertEqual(result_path, target_path)
            warp_raster.assert_called_once()
            warp_args, warp_kwargs = warp_raster.call_args
            self.assertEqual(warp_args[0], str(condition_path))
            self.assertEqual(warp_args[1], reference_info["pixel_size"])
            self.assertEqual(warp_kwargs["target_bb"], reference_info["bounding_box"])
            self.assertEqual(
                warp_kwargs["target_projection_wkt"],
                reference_info["projection_wkt"],
            )
            self.assertEqual(
                warp_kwargs["vector_mask_options"],
                {"mask_vector_path": str(vector_path)},
            )

            raster_calculator.assert_called_once()
            calc_args, calc_kwargs = raster_calculator.call_args
            self.assertEqual(calc_args[2], str(target_path))
            self.assertEqual(calc_args[3], workflow_runner.gdal.GDT_Byte)
            self.assertEqual(calc_args[4], 0)
            self.assertFalse(calc_kwargs["calc_raster_stats"])


if __name__ == "__main__":
    unittest.main()
