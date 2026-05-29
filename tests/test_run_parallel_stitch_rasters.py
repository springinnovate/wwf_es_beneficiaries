import tempfile
import unittest
from pathlib import Path

from utils.run_parallel_stitch_rasters import (
    build_stitch_jobs,
    expand_raster_list_args,
)


class RunParallelStitchRastersTests(unittest.TestCase):

    def test_expands_globs_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            first_path = workspace_path / "a.txt"
            second_path = workspace_path / "b.txt"
            first_path.write_text("a.tif\n", encoding="utf-8")
            second_path.write_text("b.tif\n", encoding="utf-8")

            result = expand_raster_list_args(
                [str(first_path), str(workspace_path / "*.txt")]
            )

            self.assertEqual(result, [first_path.resolve(), second_path.resolve()])

    def test_builds_default_output_paths_next_to_inputs(self):
        with tempfile.TemporaryDirectory() as workspace:
            raster_list_path = Path(workspace) / "example.txt"

            jobs = build_stitch_jobs([raster_list_path], None, "")

            self.assertEqual(jobs[0].raster_list_path, raster_list_path.resolve())
            self.assertEqual(
                jobs[0].output_raster_path,
                raster_list_path.with_suffix(".tif").resolve(),
            )

    def test_builds_output_paths_in_output_dir_with_suffix(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            raster_list_path = workspace_path / "example.txt"
            output_dir = workspace_path / "stitched"

            jobs = build_stitch_jobs([raster_list_path], output_dir, "_stitched")

            self.assertEqual(
                jobs[0].output_raster_path,
                (output_dir / "example_stitched.tif").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
