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
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from typing import Sequence


DEFAULT_STATUS_INTERVAL_SECONDS = 30
PROGRESS_EVENT_PREFIX = "STITCH_PROGRESS "

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected but not required.
    tqdm = None


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


def _build_stitch_command(
    job: StitchJob,
    stitcher_script_path: Path,
    python_executable: str,
    nodata: str | None = None,
) -> list[str]:
    """Build the subprocess command for one stitch job."""
    command = [
        python_executable,
        os.fspath(stitcher_script_path),
        os.fspath(job.raster_list_path),
        os.fspath(job.output_raster_path),
        "--progress-json",
    ]
    if nodata is not None:
        command.extend(["--nodata", nodata])
    return command


def run_stitch_job(
    job: StitchJob,
    stitcher_script_path: Path,
    python_executable: str,
    progress_position: int | None = None,
    nodata: str | None = None,
) -> StitchResult:
    """Run one stitch job as a child Python process.

    Args:
        job: Stitch job to execute.
        stitcher_script_path: Path to ``high_performance_stitch_rasters.py``.
        python_executable: Python executable used to launch the stitcher.
        progress_position: Optional tqdm line position for this job's live
            progress bar.
        nodata: Optional nodata override to pass to the stitcher.

    Returns:
        ``StitchResult`` with captured output and runtime.
    """
    start_time = time.monotonic()
    process = subprocess.Popen(
        _build_stitch_command(
            job,
            stitcher_script_path,
            python_executable,
            nodata,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    progress_bar = None
    if tqdm is not None and progress_position is not None:
        progress_bar = tqdm(
            total=0,
            desc=f"{job.raster_list_path.stem}: starting",
            unit="step",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
        )
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            if line.startswith(PROGRESS_EVENT_PREFIX):
                event = json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
                if progress_bar is not None:
                    _update_job_progress_bar(progress_bar, job, event)
                continue
            output_lines.append(line)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    returncode = process.wait()
    return StitchResult(
        job=job,
        returncode=returncode,
        stdout="\n".join(output_lines),
        stderr="",
        elapsed_seconds=time.monotonic() - start_time,
    )


def _short_progress_text(text: str, max_length: int = 34) -> str:
    """Return a compact label for a tqdm progress row."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "..."


def _update_job_progress_bar(
    progress_bar,
    job: StitchJob,
    event: dict,
) -> None:
    """Apply one child-process progress event to a job progress bar."""
    event_type = event.get("event")
    if event_type == "stage_start":
        stage = _short_progress_text(event["stage"])
        progress_bar.reset(total=event.get("total", 0))
        progress_bar.n = 0
        progress_bar.unit = event.get("unit", "item")
        progress_bar.set_description(
            f"{_short_progress_text(job.raster_list_path.stem, 24)}: {stage}"
        )
        progress_bar.refresh()
    elif event_type == "progress":
        progress_bar.update(event.get("advance", 1))
    elif event_type == "stage_done":
        if progress_bar.total is not None and progress_bar.n < progress_bar.total:
            progress_bar.update(progress_bar.total - progress_bar.n)


def run_stitch_jobs(
    jobs: Sequence[StitchJob],
    workers: int,
    stitcher_script_path: Path,
    python_executable: str,
    status_interval_seconds: int = DEFAULT_STATUS_INTERVAL_SECONDS,
    nodata: str | None = None,
) -> list[StitchResult]:
    """Run stitch jobs in parallel with a worker limit.

    Args:
        jobs: Stitch jobs to execute.
        workers: Maximum number of jobs to run at the same time.
        stitcher_script_path: Path to ``high_performance_stitch_rasters.py``.
        python_executable: Python executable used to launch each job.
        status_interval_seconds: Seconds between status messages while jobs
            are still running.
        nodata: Optional nodata override to pass to each stitch job.

    Returns:
        Ordered list of completed ``StitchResult`` objects in completion order.

    Raises:
        ValueError: If ``workers`` is less than 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    results: list[StitchResult] = []
    futures_to_jobs = {}
    positions = queue.Queue()
    for position in range(1, min(workers, len(jobs)) + 1):
        positions.put(position)
    start_time = time.monotonic()
    last_status_time = start_time
    overall_progress_bar = None
    if tqdm is not None:
        overall_progress_bar = tqdm(
            total=len(jobs),
            desc="stitch jobs",
            unit="job",
            position=0,
            dynamic_ncols=True,
        )

    def run_job_with_progress_position(job: StitchJob) -> StitchResult:
        position = positions.get()
        try:
            return run_stitch_job(
                job,
                stitcher_script_path,
                python_executable,
                progress_position=position,
                nodata=nodata,
            )
        finally:
            positions.put(position)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for job in jobs:
                future = executor.submit(run_job_with_progress_position, job)
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
                    if overall_progress_bar is not None:
                        overall_progress_bar.update(1)
                    status = "ok" if result.returncode == 0 else "failed"
                    status_message = (
                        f"[{len(results)}/{len(jobs)}] {status}: "
                        f"{result.job.raster_list_path.name} "
                        f"({result.elapsed_seconds:.1f}s)"
                    )
                    if tqdm is not None:
                        tqdm.write(status_message)
                    else:
                        print(status_message)
                    if result.stdout:
                        if tqdm is not None:
                            tqdm.write(result.stdout.rstrip())
                        else:
                            print(result.stdout.rstrip())
                    if result.stderr:
                        if tqdm is not None:
                            tqdm.write(result.stderr.rstrip(), file=sys.stderr)
                        else:
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
                    status_message = (
                        f"still running {len(pending_futures)} job(s) after "
                        f"{now - start_time:.0f}s: {', '.join(running_names)}"
                    )
                    if tqdm is not None:
                        tqdm.write(status_message)
                    else:
                        print(status_message)
                    last_status_time = now
    finally:
        if overall_progress_bar is not None:
            overall_progress_bar.close()

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
    parser.add_argument(
        "--nodata",
        default=None,
        help=(
            "Use this nodata value when the first raster in a stitch job does "
            "not define one, and ignore this value in every input raster."
        ),
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
        args.nodata,
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
