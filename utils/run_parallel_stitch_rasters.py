"""Run multiple raster stitching jobs with a fixed worker limit.

This script launches ``high_performance_stitch_rasters.py`` as child Python
processes. Each input ``.txt`` file is treated as one stitch job, and the
default output path is the same path with ``.tif`` as the suffix.

Example:
    python utils/run_parallel_stitch_rasters.py --workers 12 "D:\\frontiers_data\\countries\\*.txt"
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import glob
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


DEFAULT_STATUS_INTERVAL_SECONDS = 30


@dataclass(frozen=True)
class StitchJob:
    """Describe one stitcher subprocess invocation.

    Args:
        raster_list_path: Path to the text file listing rasters to stitch.
        output_raster_path: Path where the stitched GeoTIFF should be written.
    """

    raster_list_path: Path
    output_raster_path: Path


@dataclass(frozen=True)
class StitchResult:
    """Describe the outcome of one stitcher subprocess.

    Args:
        job: Stitch job that was executed.
        returncode: Subprocess return code.
        stdout: Captured standard output.
        stderr: Captured standard error.
        elapsed_seconds: Job runtime in seconds.
    """

    job: StitchJob
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def expand_raster_list_args(raster_list_args: Sequence[str]) -> list[Path]:
    """Expand raster-list file arguments and glob patterns.

    Args:
        raster_list_args: Command-line arguments. Each argument may be a
            concrete path or a glob pattern.

    Returns:
        Ordered, de-duplicated list of raster-list paths.

    Raises:
        ValueError: If no files are found or a glob pattern does not match.
        FileNotFoundError: If a concrete path does not exist.
    """
    raster_list_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raster_list_arg in raster_list_args:
        has_glob = any(char in raster_list_arg for char in "*?[")
        if has_glob:
            matches = sorted(glob.glob(raster_list_arg))
            if not matches:
                raise ValueError(f"No files matched glob: {raster_list_arg}")
            candidate_paths = [Path(match) for match in matches]
        else:
            candidate_path = Path(raster_list_arg)
            if not candidate_path.exists():
                raise FileNotFoundError(candidate_path)
            candidate_paths = [candidate_path]

        for candidate_path in candidate_paths:
            resolved_path = candidate_path.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            raster_list_paths.append(resolved_path)

    if not raster_list_paths:
        raise ValueError("No raster-list files were provided.")
    return raster_list_paths


def build_stitch_jobs(
    raster_list_paths: Sequence[Path],
    output_dir: Path | None,
    output_suffix: str,
) -> list[StitchJob]:
    """Build stitch jobs from input list files.

    Args:
        raster_list_paths: Paths to text files listing rasters.
        output_dir: Optional directory where all stitched rasters should be
            written. If ``None``, each output is written beside its input list.
        output_suffix: Suffix inserted between each list-file stem and
            ``.tif``. For example, ``"_stitched"`` creates
            ``name_stitched.tif``.

    Returns:
        Ordered list of ``StitchJob`` instances.
    """
    jobs = []
    for raster_list_path in raster_list_paths:
        target_dir = Path(output_dir).resolve() if output_dir else raster_list_path.parent
        output_raster_path = target_dir / f"{raster_list_path.stem}{output_suffix}.tif"
        jobs.append(
            StitchJob(
                raster_list_path=raster_list_path.resolve(),
                output_raster_path=output_raster_path.resolve(),
            )
        )
    return jobs


def run_stitch_job(
    job: StitchJob,
    stitcher_script_path: Path,
    python_executable: str,
) -> StitchResult:
    """Run one stitch job as a child Python process.

    Args:
        job: Stitch job to execute.
        stitcher_script_path: Path to ``high_performance_stitch_rasters.py``.
        python_executable: Python executable used to launch the stitcher.

    Returns:
        ``StitchResult`` with captured output and runtime.
    """
    start_time = time.monotonic()
    completed_process = subprocess.run(
        [
            python_executable,
            os.fspath(stitcher_script_path),
            os.fspath(job.raster_list_path),
            os.fspath(job.output_raster_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return StitchResult(
        job=job,
        returncode=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
        elapsed_seconds=time.monotonic() - start_time,
    )


def run_stitch_jobs(
    jobs: Sequence[StitchJob],
    workers: int,
    stitcher_script_path: Path,
    python_executable: str,
    status_interval_seconds: int = DEFAULT_STATUS_INTERVAL_SECONDS,
) -> list[StitchResult]:
    """Run stitch jobs in parallel with a worker limit.

    Args:
        jobs: Stitch jobs to execute.
        workers: Maximum number of jobs to run at the same time.
        stitcher_script_path: Path to ``high_performance_stitch_rasters.py``.
        python_executable: Python executable used to launch each job.
        status_interval_seconds: Seconds between status messages while jobs
            are still running.

    Returns:
        Ordered list of completed ``StitchResult`` objects in completion order.

    Raises:
        ValueError: If ``workers`` is less than 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    results: list[StitchResult] = []
    futures_to_jobs = {}
    start_time = time.monotonic()
    last_status_time = start_time

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for job in jobs:
            future = executor.submit(
                run_stitch_job,
                job,
                stitcher_script_path,
                python_executable,
            )
            futures_to_jobs[future] = job

        pending_futures = set(futures_to_jobs)
        while pending_futures:
            done_futures, pending_futures = wait(
                pending_futures,
                timeout=1,
                return_when=FIRST_COMPLETED,
            )

            for future in done_futures:
                result = future.result()
                results.append(result)
                status = "ok" if result.returncode == 0 else "failed"
                print(
                    f"[{len(results)}/{len(jobs)}] {status}: "
                    f"{result.job.raster_list_path.name} "
                    f"({result.elapsed_seconds:.1f}s)"
                )
                if result.stdout:
                    print(result.stdout.rstrip())
                if result.stderr:
                    print(result.stderr.rstrip(), file=sys.stderr)

            now = time.monotonic()
            if (
                pending_futures
                and status_interval_seconds > 0
                and now - last_status_time >= status_interval_seconds
            ):
                running_names = [
                    futures_to_jobs[future].raster_list_path.name
                    for future in pending_futures
                ]
                print(
                    f"still running {len(pending_futures)} job(s) after "
                    f"{now - start_time:.0f}s: {', '.join(running_names)}"
                )
                last_status_time = now

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser for the parallel stitch runner.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run high_performance_stitch_rasters.py for many raster-list text "
            "files with a fixed worker limit."
        )
    )
    parser.add_argument(
        "raster_lists",
        nargs="+",
        help=(
            "Raster-list text files or glob patterns. Each list becomes one "
            "stitched GeoTIFF."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of stitch jobs to run at once. Default: 4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for stitched rasters. Defaults to each list "
            "file's directory."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Suffix inserted before .tif in output names. Example: "
            "--output-suffix _stitched creates name_stitched.tif."
        ),
    )
    parser.add_argument(
        "--stitcher-script",
        type=Path,
        default=Path(__file__).with_name("high_performance_stitch_rasters.py"),
        help="Path to high_performance_stitch_rasters.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child stitch processes.",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=DEFAULT_STATUS_INTERVAL_SECONDS,
        help="Seconds between running-job status messages. Use 0 to disable.",
    )
    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    raster_list_paths = expand_raster_list_args(args.raster_lists)
    jobs = build_stitch_jobs(
        raster_list_paths,
        args.output_dir,
        args.output_suffix,
    )
    print(f"running {len(jobs)} stitch job(s) with {args.workers} worker(s)")

    results = run_stitch_jobs(
        jobs,
        args.workers,
        args.stitcher_script.resolve(),
        args.python,
        args.status_interval,
    )
    failed_results = [result for result in results if result.returncode != 0]
    if failed_results:
        failed_names = ", ".join(
            result.job.raster_list_path.name for result in failed_results
        )
        raise SystemExit(f"{len(failed_results)} stitch job(s) failed: {failed_names}")
    print("all stitch jobs completed successfully")


if __name__ == "__main__":
    main()
