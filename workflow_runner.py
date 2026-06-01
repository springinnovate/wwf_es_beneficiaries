"""Runner for gathering ds pop info given masks.

To run in docker environment:
- windows:
docker build -t therealspring/ds_beneficiaries:latest . && docker run --rm -it -v "%CD%":/usr/local/wwf_es_beneficiaries therealspring/ds_beneficiaries:latest
- linux/mac:
docker build -t ds_beneficiaries:latest . && docker run --rm -it -v `pwd`:/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest

To run the script:

python workflow_runner.py ./example_roadmap2030_pop_downstream_analysis.yaml
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import collections
import contextlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import argparse
import os
import glob
import logging
import sys
import threading
import time
from itertools import islice

from ecoshard import taskgraph
import psutil
import yaml

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from ecoshard import geoprocessing
from ecoshard.geoprocessing import routing
from osgeo import gdal
from shapely.geometry import box
from shapely.ops import transform
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
import rasterio.features
import pyogrio
from pyproj import CRS, Transformer, Geod
from tqdm.auto import tqdm

import shortest_distances

RASTER_BLOCK_SIZE = 256
HEARTBEAT_INTERVAL_SECONDS = 60
DEFAULT_GDAL_CACHEMAX_MB = 128
DEFAULT_CONDITIONAL_TASKGRAPH_WORKERS = 2
DEFAULT_TRAVEL_TIME_TASKGRAPH_WORKERS = 4
TRAVEL_TIME_MAX_DISTANCE_M_PER_HOUR = 104_000
TRAVEL_TIME_MAX_WINDOW_BYTES = 2 * 1024**3
POPULATION_COMBINE_VRT_NODATA = -1
DISTANCE_TRANSFORM_NODATA = -1
DOWNSTREAM_COVERAGE_EPSILON = 100 * np.finfo(float).eps
GTIFF_CREATION_OPTIONS = (
    "TILED=YES",
    "BIGTIFF=YES",
    "COMPRESS=LZW",
    f"BLOCKXSIZE={RASTER_BLOCK_SIZE}",
    f"BLOCKYSIZE={RASTER_BLOCK_SIZE}",
    "NUM_THREADS=ALL_CPUS",
)
GTIFF_CREATION_TUPLE = ("GTIFF", GTIFF_CREATION_OPTIONS)


def _format_bytes(size_bytes: int | float) -> str:
    """Return a compact human-readable byte size."""
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}"
        size /= 1024


def configure_gdal_cache(cachemax_mb: int | None = None) -> int:
    """Set GDAL's per-process cache limit and return it in MiB."""
    if cachemax_mb is None:
        cachemax_mb = int(os.environ.get("GDAL_CACHEMAX", DEFAULT_GDAL_CACHEMAX_MB))
    cachemax_mb = int(cachemax_mb)
    os.environ["GDAL_CACHEMAX"] = str(cachemax_mb)
    cachemax_bytes = cachemax_mb * 1024 * 1024
    if hasattr(gdal, "SetCacheMax64"):
        gdal.SetCacheMax64(cachemax_bytes)
    else:
        gdal.SetCacheMax(cachemax_bytes)
    return cachemax_mb


def _process_tree_rss_bytes() -> int:
    """Return RSS for this process and any living child processes."""
    total = 0
    process_list = [psutil.Process(os.getpid())]
    try:
        process_list.extend(process_list[0].children(recursive=True))
    except psutil.Error:
        pass
    for process in process_list:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            continue
    return total


def _memory_status_text() -> str:
    """Return current system and workflow memory use for heartbeat logs."""
    virtual_memory = psutil.virtual_memory()
    return (
        "memory: "
        f"available={_format_bytes(virtual_memory.available)}, "
        f"used={virtual_memory.percent:.1f}%, "
        f"workflow_rss={_format_bytes(_process_tree_rss_bytes())}"
    )


@contextlib.contextmanager
def _log_memory_scope(logger: logging.Logger, label: str):
    """Log process-tree memory before and after a diagnostic scope."""
    start_time = time.monotonic()
    start_rss = _process_tree_rss_bytes()
    logger.info("%s started; %s", label, _memory_status_text())
    try:
        yield
    finally:
        end_rss = _process_tree_rss_bytes()
        elapsed_seconds = time.monotonic() - start_time
        logger.info(
            "%s finished in %.1fs; rss_delta=%s; %s",
            label,
            elapsed_seconds,
            _format_bytes(end_rss - start_rss),
            _memory_status_text(),
        )


def _set_tiled_geotiff_creation_options(raster_meta: dict) -> None:
    """Set Rasterio creation options for block-aligned GeoTIFF writes."""
    raster_meta.update(
        {
            "tiled": True,
            "blockxsize": RASTER_BLOCK_SIZE,
            "blockysize": RASTER_BLOCK_SIZE,
            "compress": "lzw",
            "BIGTIFF": "YES",
        }
    )


@contextlib.contextmanager
def _log_heartbeat(
    logger: logging.Logger,
    message_factory: Callable[[], str],
    interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
):
    """Emit low-frequency heartbeat logs while a blocking operation runs."""
    start_time = time.monotonic()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(interval_seconds):
            logger.info(
                "%s; elapsed=%ds",
                message_factory(),
                int(time.monotonic() - start_time),
            )

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)


QUIET_LOGGER_NAMES = [
    "ecoshard",
    "ecoshard.taskgraph",
    "pyogrio",
    "pyogrio._io",
    "geopandas",
    "rasterio",
]

APP_LOGGER_NAMES = [
    __name__,
    "workflow_runner",
    "shortest_distances",
]

for logger_name in QUIET_LOGGER_NAMES:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

FULL_RASTER_EXTENT_AOI_ID = "full_raster_extent"


class TqdmLoggingHandler(logging.StreamHandler):
    """Write console logs without corrupting active tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


@dataclass
class PickedCRS:
    """Used to define a return type for CRS."""

    crs: CRS
    rationale: str


def _normalize_lon(lon: float) -> float:
    """Make sure long maps to -180/180."""
    return ((lon + 180) % 360) - 180


def _centroid_lon(minx: float, maxx: float) -> float:
    """Make sure if the meridian is 360 we handle that."""
    left, right = _normalize_lon(minx), _normalize_lon(maxx)
    if right < left:  # wrapped
        right += 360
    mid = (right + left) / 2.0
    return _normalize_lon(mid)


def create_circular_kernel(kernel_path, buffer_size_in_px):
    diameter = buffer_size_in_px * 2 + 1
    kernel_array = np.zeros((diameter, diameter), dtype=np.float32)
    cx, cy = buffer_size_in_px, buffer_size_in_px

    for i in range(diameter):
        for j in range(diameter):
            if (i - cx) ** 2 + (j - cy) ** 2 <= buffer_size_in_px**2:
                kernel_array[i, j] = 1.0

    driver = gdal.GetDriverByName("GTiff")
    out_raster = driver.Create(
        kernel_path,
        diameter,
        diameter,
        1,
        gdal.GDT_Float32,
        options=GTIFF_CREATION_OPTIONS,
    )
    out_raster.GetRasterBand(1).WriteArray(kernel_array)
    out_raster.FlushCache()
    out_raster = None


def choose_equidistant_crs_from_bbox(aoi_vector_path: str) -> PickedCRS:
    """Pick a good CRS that is equadistant depending on size.

    Args:
        minx,miny,maxx,maxy: geographic bbox in WGS84 degrees.

    Returns:
        PickedCRS with a pyproj.CRS and a rationale string.
    """
    aoi_gdf = gpd.read_file(aoi_vector_path)
    aoi_gdf = aoi_gdf.set_geometry(aoi_gdf.geometry.make_valid())
    if aoi_gdf.crs and aoi_gdf.crs.to_string() != "EPSG:4326":
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = aoi_gdf.total_bounds

    # Normalize
    minx_n, maxx_n = _normalize_lon(minx), _normalize_lon(maxx)
    # Handle wrap for span
    if maxx_n < minx_n:
        maxx_span = maxx_n + 360
        minx_span = minx_n
    else:
        minx_span, maxx_span = minx_n, maxx_n

    lon_span_deg = maxx_span - minx_span
    lat_span_deg = maxy - miny
    lon0 = _centroid_lon(minx, maxx)
    lat0 = (miny + maxy) / 2.0

    # First, get the rough size in km using geodesic projection
    geod = Geod(ellps="WGS84")
    ew_km = abs(geod.line_length([minx, maxx], [lat0, lat0])) / 1000.0
    ns_km = abs(geod.line_length([lon0, lon0], [miny, maxy])) / 1000.0
    max_span_km = max(ew_km, ns_km)

    # Choose UTM if within a single zone and not too close to poles
    if lon_span_deg <= 6.0 and abs(lat0) < 84.0:
        zone = int((lon0 + 180) // 6) + 1
        # little trick to reverse engineer the epsg code
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
        return PickedCRS(
            crs=CRS.from_epsg(epsg),
            rationale=(
                f"UTM zone {zone} (EPSG:{epsg}) - bbox spans "
                f"~{lon_span_deg:.1f} degrees, {max_span_km:.0f} km"
            ),
        )

    # Regional if <= ~2500 km, asmuth equadistant centered on bbox centroid
    if max_span_km <= 2500:
        crs = CRS.from_proj4(
            (
                f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +x_0=0 +y_0=0 "
                f"+datum=WGS84 +units=m +no_defs"
            )
        )
        return PickedCRS(
            crs=crs,
            rationale=(
                f"Azimuthal Equidistant centered at ({lat0:.4f}, {lon0:.4f}) "
                f"- preserves distances from center; span {max_span_km:.0f} "
                f"km"
            ),
        )

    # big problem if very wide EW, use Equidistant Conic with parallels
    # near lower/upper lat
    if ew_km > ns_km:
        lat1 = miny + lat_span_deg * 0.25
        lat2 = miny + lat_span_deg * 0.75
        crs = CRS.from_proj4(
            f"+proj=eqdc +lat_1={lat1:.8f} +lat_2={lat2:.8f} "
            f"+lat_0={lat0:.8f} +lon_0={lon0:.8f} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        return PickedCRS(
            crs=crs,
            rationale=(
                f"Equidistant Conic (lat1={lat1:.2f}, lat2={lat2:.2f}) "
                f"- large E–W extent {ew_km:.0f} km"
            ),
        )

    # fallback, AEQD for very tall regions
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +x_0=0 +y_0=0 "
        f"+datum=WGS84 +units=m +no_defs"
    )
    return PickedCRS(
        crs=crs,
        rationale=(
            f"Azimuthal Equidistant fallback - " f"large N–S extent {ns_km:.0f} km"
        ),
    )


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def process_config(config_path: Path) -> Dict[str, Any]:
    """Parse and validate a YAML file for the beneficiaries workflow.

    This function reads a YAML file, enforces basic correctness checks,
    normalizes shapes, and returns a dictionary with canonical keys used by the
    pipeline. It verifies that the YAML's 'run_name' matches the configuration
    filename stem, that required inputs are present, and that the 'sections'
    block contains both a
    'masks' and a 'combine' section with the expected structures.

    Args:
        config_path (Path): Path to the YAML configuration file. The filename
            stem must equal the 'run_name' value inside the YAML (e.g., a file
            named 'my_run.yaml' must contain 'run_name: my_run').

    Returns:
        Dict[str, Any]: A normalized configuration dictionary with keys:
            - 'run_name' (str)
            - 'work_dir' (str)
            - 'output_dir' (str)
            - 'inputs' (dict):
                - 'population_raster_path' (str)
                - 'traveltime_raster_path' (str)
                - 'subwatershed_vector_path' (str)
                - 'aoi_vector_pattern' (list[str])
                - 'analyze_full_raster_extent' (bool)
                - 'debug_drain_index' (int | None)
                - 'taskgraph_workers' (int | None)
                - 'gdal_cachemax_mb' (int)
            - 'masks' (list[dict]): Each item has 'id' (str), 'type' (str),
              and 'params' (dict).
            - 'combine' (list[dict]): As provided in the YAML 'combine'
               section.
            - 'logging' (dict): Contains 'level' (str) and 'to_file' (str).

    Raises:
        ValueError: If any of the following occur:
            - 'run_name' does not match the configuration filename stem.
            - The 'inputs' section is missing or empty.
            - Any required input is missing:
                'population_raster_path', 'traveltime_raster_path',
                'subwatershed_vector_path', or 'aoi_vector_pattern' when
                'analyze_full_raster_extent' is not enabled.
            - Any 'sections' entry is not a mapping (dict).
            - A 'masks' or 'combine' item within 'sections' is not a dict.
            - The 'sections' block does not include both a 'masks' and
              a 'combine' section.
            - Aggregated structural errors are detected during validation.

    Notes:
        - The function only reads from disk (the YAML file). It does not touch
          or validate the existence of referenced data paths here.
        - 'aoi_vector_pattern' is normalized to a list via a helper
           like '_as_list'.
    """
    with config_path.open("r", encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f) or {}
    run_name = raw_yaml.get("run_name", "")

    if run_name != config_path.stem:
        raise ValueError(
            f"The `run_name` ({run_name}) does not match the configuration  "
            f"filename ({config_path.stem}). This check helps catch copy-paste "
            f"mistakes or using the wrong config file, the two should be "
            f"identical to avoid confusion."
        )
    work_dir = raw_yaml.get("work_dir", "")
    output_dir = raw_yaml.get("output_dir", "")

    inputs = raw_yaml.get("inputs", {}) or {}
    if not inputs:
        raise ValueError("missing `inputs` section, cannot continue")
    population_raster_path = inputs.get("population_raster_path", "")
    traveltime_raster_path = inputs.get("traveltime_raster_path", "")
    subwatershed_vector_path = inputs.get("subwatershed_vector_path", "")
    dem_raster_path = inputs.get("dem_raster_path", "")
    aoi_vector_pattern = [
        pattern for pattern in _as_list(inputs.get("aoi_vector_pattern", [])) if pattern
    ]
    analyze_full_raster_extent = inputs.get("analyze_full_raster_extent", False)
    if not isinstance(analyze_full_raster_extent, bool):
        raise ValueError(
            "`inputs.analyze_full_raster_extent` must be true or false, "
            f"got {analyze_full_raster_extent!r}"
        )
    debug_drain_index = inputs.get("debug_drain_index", None)
    if debug_drain_index is not None:
        if isinstance(debug_drain_index, bool) or not isinstance(
            debug_drain_index, int
        ):
            raise ValueError(
                "`inputs.debug_drain_index` must be a non-negative integer, "
                f"got {debug_drain_index!r}"
            )
        if debug_drain_index < 0:
            raise ValueError(
                "`inputs.debug_drain_index` must be a non-negative integer, "
                f"got {debug_drain_index}"
            )

    taskgraph_workers = inputs.get("taskgraph_workers", None)
    if taskgraph_workers is not None:
        if isinstance(taskgraph_workers, bool) or not isinstance(
            taskgraph_workers, int
        ):
            raise ValueError(
                "`inputs.taskgraph_workers` must be a positive integer, "
                f"got {taskgraph_workers!r}"
            )
        if taskgraph_workers < 1:
            raise ValueError(
                "`inputs.taskgraph_workers` must be a positive integer, "
                f"got {taskgraph_workers}"
            )

    gdal_cachemax_mb = inputs.get("gdal_cachemax_mb", DEFAULT_GDAL_CACHEMAX_MB)
    if isinstance(gdal_cachemax_mb, bool) or not isinstance(gdal_cachemax_mb, int):
        raise ValueError(
            "`inputs.gdal_cachemax_mb` must be a positive integer, "
            f"got {gdal_cachemax_mb!r}"
        )
    if gdal_cachemax_mb < 1:
        raise ValueError(
            "`inputs.gdal_cachemax_mb` must be a positive integer, "
            f"got {gdal_cachemax_mb}"
        )

    wgs84_pixel_size = inputs.get("wgs84_pixel_size", None)
    travel_time_pixel_size_m = inputs.get("travel_time_pixel_size_m", None)
    buffer_size_m = inputs.get("buffer_size_m", None)

    missing_messages = []
    if not population_raster_path:
        missing_messages.append(f"population_raster_path {population_raster_path}")
    elif not Path(population_raster_path).exists():
        missing_messages.append(
            f"population_raster_path does not exist: {population_raster_path}"
        )

    if not traveltime_raster_path:
        missing_messages.append(f"traveltime_raster_path {traveltime_raster_path}")
    elif not Path(traveltime_raster_path).exists():
        missing_messages.append(
            f"traveltime_raster_path does not exist: {traveltime_raster_path}"
        )

    if not dem_raster_path:
        missing_messages.append("dem_raster_path (path to DEM raster)")
    elif not Path(dem_raster_path).exists():
        missing_messages.append(f"dem_raster_path does not exist: {dem_raster_path}")

    if not subwatershed_vector_path:
        missing_messages.append(f"subwatershed_vector_path {subwatershed_vector_path}")
    elif not Path(subwatershed_vector_path).exists():
        missing_messages.append(
            f"subwatershed_vector_path does not exist: {subwatershed_vector_path}"
        )

    if analyze_full_raster_extent and aoi_vector_pattern:
        missing_messages.append(
            "aoi_vector_pattern must be empty when "
            "analyze_full_raster_extent is true"
        )
    elif not analyze_full_raster_extent and not aoi_vector_pattern:
        missing_messages.append("aoi_vector_pattern is empty")

    if wgs84_pixel_size is None:
        missing_messages.append(
            "wgs84_pixel_size (pixel size of population raster in degrees)"
        )

    if travel_time_pixel_size_m is None:
        missing_messages.append(
            "travel_time_pixel_size_m (pixel size of travel-time raster in meters)"
        )

    if buffer_size_m is None:
        missing_messages.append(
            "buffer_size_m (buffer size in meters for AOI expansion)"
        )

    if missing_messages:
        msg = "Missing required input(s):\n  - " + "\n  - ".join(missing_messages)
        raise ValueError(msg)

    sections = raw_yaml.get("sections", []) or []
    masks: List[Dict[str, Any]] = []
    combine_logic: List[Dict[str, Any]] = []

    errors = []
    found_sections = []
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            errors.append(
                f"sections[{idx}] must be a mapping (dict), got "
                f"{type(section).__name__}"
            )
            continue
        matched = False
        if "masks" in section:
            found_sections.append("masks")
            matched = True
            for jdx, m in enumerate(_as_list(section.get("masks", []))):
                if not isinstance(m, dict):
                    errors.append(
                        f"sections[{idx}].masks[{jdx}] must be a mapping "
                        f"(dict), got {type(m).__name__}"
                    )
                    continue
                masks.append(
                    {
                        "id": m.get("id", ""),
                        "type": m.get("type", ""),
                        "params": m.get("params", {}) or {},
                    }
                )

        if "combine" in section:
            found_sections.append("combine")
            matched = True
            for kdx, c in enumerate(_as_list(section.get("combine", []))):
                if not isinstance(c, dict):
                    errors.append(
                        f"sections[{idx}].combine[{kdx}] must be a mapping "
                        f"(dict), got {type(c).__name__}"
                    )
                    continue
                combine_logic.append(c)

        if not matched:
            errors.append(
                f"sections[{idx}] must contain at least one of " f'["masks", "combine"]'
            )
    if set(found_sections) != set(["masks", "combine"]):
        raise ValueError(
            "Expected both a `masks` and `combine` section but missing at least one."
        )

    if errors:
        raise ValueError("Invalid sections:\n  - " + "\n  - ".join(errors))
    if debug_drain_index is not None and not any(
        mask_section.get("type") == "conditional_raster" for mask_section in masks
    ):
        raise ValueError(
            "`inputs.debug_drain_index` can only be used when at least one "
            "conditional_raster mask is configured."
        )
    logging_cfg = raw_yaml.get("logging", {}) or {}
    log_level = logging_cfg.get("level", "INFO")
    log_to_file = logging_cfg.get("to_file", "")

    return {
        "run_name": run_name,
        "work_dir": work_dir,
        "output_dir": output_dir,
        "inputs": {
            "population_raster_path": Path(population_raster_path),
            "traveltime_raster_path": Path(traveltime_raster_path),
            "subwatershed_vector_path": Path(subwatershed_vector_path),
            "dem_raster_path": Path(dem_raster_path),
            "aoi_vector_pattern": aoi_vector_pattern,
            "analyze_full_raster_extent": analyze_full_raster_extent,
            "debug_drain_index": debug_drain_index,
            "taskgraph_workers": taskgraph_workers,
            "gdal_cachemax_mb": gdal_cachemax_mb,
            "wgs84_pixel_size": float(wgs84_pixel_size),
            "travel_time_pixel_size_m": float(travel_time_pixel_size_m),
            "buffer_size_m": float(buffer_size_m),
        },
        "masks": masks,
        "combine": combine_logic,
        "logging": {
            "level": log_level,
            "to_file": log_to_file,
        },
    }


def setup_logger(level: str, log_file: str) -> logging.Logger:
    """Configure and return a logger for the analysis pipeline.

    This function creates a logger named __name__ with the given
    log level and attaches two handlers:
      * A stream handler that writes to stdout.
      * A file handler that writes to the specified file, if provided.

    Both handlers use a formatter that includes timestamp, log level,
    filename, line number, and the log message.

    Args:
        level (str): Logging level (``"DEBUG"``, ``"INFO"``, etc).
        log_file (str): Path to the log file. If empty, no file handler is
        added.

    Returns:
        logging.Logger: Configured logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers.clear()

    fmt = "%(asctime)s %(filename)s:%(lineno)d [%(levelname)s]  %(message)s"

    sh = TqdmLoggingHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        root_logger.addHandler(fh)

    for logger_name in APP_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(level.upper())
    for logger_name in QUIET_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return root_logger


def validate_paths(config: Dict[str, Any]) -> None:
    """Validate that required file paths exist in the configuration.

    This function checks for the presence and existence of file paths in the
    configuration dictionary. Glob patterns are skipped from existence checks.
    All issues are collected, and if any are found, a single ``ValueError`` is
    raised with a summary of the problems.

    Args:
        config (Dict[str, Any]): Parsed configuration dictionary.

    Raises:
        ValueError: If one or more required paths are missing or do not exist.
    """
    issues: List[Tuple[str, str]] = []

    def _check(path_like: Any, label: str) -> None:
        if not path_like:
            issues.append((label, "missing"))
            return
        path_str = os.fspath(path_like)
        if any(ch in path_str for ch in ["*", "?", "["]):
            # skip globs
            return
        if not os.path.exists(path_str):
            issues.append((label, f"not found: {path_str}"))

    inputs = config.get("inputs", {})
    for label in [
        "population_raster_path",
        "traveltime_raster_path",
        "subwatershed_vector_path",
    ]:
        _check(inputs.get(label), label)

    for i, mask_section in enumerate(config.get("masks", [])):
        params = mask_section.get("params", {}) or {}
        for key, val in params.items():
            if key.endswith("_path"):
                _check(val, f"mask[{i}].params.{key}")

    subwatershed_vector_path = inputs.get("subwatershed_vector_path")
    if subwatershed_vector_path and Path(subwatershed_vector_path).exists():
        try:
            subwatershed_info = pyogrio.read_info(subwatershed_vector_path)
        except Exception as error:
            issues.append(
                (
                    "subwatershed_vector_path",
                    f"could not read vector metadata: {error}",
                )
            )
        else:
            subwatershed_fields = set(subwatershed_info.get("fields", []))
            missing_fields = {
                "HYBAS_ID",
                "NEXT_DOWN",
                "NEXT_SINK",
            } - subwatershed_fields
            if missing_fields:
                issues.append(
                    (
                        "subwatershed_vector_path",
                        "missing required field(s): "
                        + ", ".join(sorted(missing_fields)),
                    )
                )

    if issues:
        formatted = "\n".join(f"- {label}: {msg}" for label, msg in issues)
        raise ValueError(
            f"Path validation failed with {len(issues)} issue(s):\n{formatted}"
        )


def print_yaml_config(config):
    """Just for debugging..."""
    logger = logging.getLogger(__name__)
    logger.debug("run_name: %s", config["run_name"])
    logger.debug("work_dir: %s", config["work_dir"])
    logger.debug("output_dir: %s", config["output_dir"])
    logger.debug("inputs:")
    for k, v in config["inputs"].items():
        if isinstance(v, list):
            logger.debug("  %s:", k)
            for item in v:
                logger.debug("    - %s", item)
        else:
            logger.debug("  %s: %s", k, v)
    logger.debug("masks:")
    for m in config["masks"]:
        logger.debug("  - id: %s", m["id"])
        logger.debug("    type: %s", m["type"])
        if m.get("params"):
            logger.debug("    params:")
            for pk, pv in m["params"].items():
                logger.debug("      %s: %s", pk, pv)
    logger.debug("combine:")
    for c in config["combine"]:
        logger.debug("  - %s", c)
    logger.debug("logging:")
    logger.debug("  level: %s", config["logging"]["level"])
    logger.debug("  to_file: %s", config["logging"]["to_file"])


def _raster_paths_for_full_extent(config: dict) -> list[Path]:
    """Return rasters that constrain the generated full-extent AOI.

    Args:
        config: Normalized workflow configuration returned by
            ``process_config``. The configuration must include input paths for
            the population, travel-time, and DEM rasters, plus any conditional
            raster mask sections that should limit the generated AOI extent.

    Returns:
        Ordered list of unique raster paths whose WGS84 bounds should be
        intersected to create the generated full-raster AOI.
    """
    inputs = config["inputs"]
    raster_paths = [
        inputs["population_raster_path"],
        inputs["traveltime_raster_path"],
        inputs["dem_raster_path"],
    ]
    for mask_section in config["masks"]:
        if mask_section["type"] != "conditional_raster":
            continue
        condition_raster_path = mask_section.get("params", {}).get(
            "condition_raster_path"
        )
        if condition_raster_path:
            raster_paths.append(Path(condition_raster_path))

    unique_paths = []
    seen_paths = set()
    for raster_path in raster_paths:
        path = Path(raster_path)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_paths.append(path)
    return unique_paths


def create_full_raster_extent_aoi(config: dict, target_aoi_vector_path: Path) -> None:
    """Create an AOI over the shared WGS84 bounds of all analysis rasters.

    Args:
        config: Normalized workflow configuration returned by
            ``process_config``. The population, travel-time, DEM, and
            conditional raster paths are used to determine the overlapping
            raster extent.
        target_aoi_vector_path: Path where the generated AOI GeoPackage should
            be written.

    Raises:
        ValueError: If any raster used to define the extent has no CRS, or if
            the rasters do not share any overlapping WGS84 bounds.
    """
    raster_paths = _raster_paths_for_full_extent(config)
    intersection_geom = None
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as raster:
            if raster.crs is None:
                raise ValueError(
                    "Raster has no CRS and cannot define an extent: " f"{raster_path}"
                )
            minx, miny, maxx, maxy = transform_bounds(
                raster.crs,
                "EPSG:4326",
                *raster.bounds,
                densify_pts=21,
            )
        raster_extent = box(minx, miny, maxx, maxy)
        if intersection_geom is None:
            intersection_geom = raster_extent
        else:
            intersection_geom = intersection_geom.intersection(raster_extent)

        if intersection_geom.is_empty:
            raise ValueError(
                "No shared raster extent remains after intersecting " f"{raster_path}."
            )

    target_aoi_vector_path.parent.mkdir(parents=True, exist_ok=True)
    aoi_gdf = gpd.GeoDataFrame(
        {"id": [FULL_RASTER_EXTENT_AOI_ID]},
        geometry=[intersection_geom],
        crs="EPSG:4326",
    )
    aoi_gdf.to_file(target_aoi_vector_path, driver="GPKG")


def collect_aoi_files(config: dict) -> dict[str, Path]:
    """Collect AOI files from the patterns.

    In config['inputs']['aoi_vector_pattern'].

    Returns:
        dict mapping file stem -> Path

    Raises:
        ValueError if two or more files share the same stem.
    """
    inputs = config.get("inputs", {})
    if inputs.get("analyze_full_raster_extent", False):
        target_aoi_vector_path = (
            Path(config["work_dir"])
            / "_generated_aois"
            / f"{FULL_RASTER_EXTENT_AOI_ID}.gpkg"
        )
        create_full_raster_extent_aoi(config, target_aoi_vector_path)
        return {FULL_RASTER_EXTENT_AOI_ID: target_aoi_vector_path.resolve()}

    patterns = inputs.get("aoi_vector_pattern", [])
    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]

    aoi_map: dict[str, Path] = {}
    for pattern in patterns:
        for p in glob.glob(pattern):
            path = Path(p).resolve()
            stem = path.stem
            if stem in aoi_map:
                raise ValueError(
                    f'Duplicate AOI stem "{stem}" found:\n'
                    f"  - {aoi_map[stem]}\n"
                    f"  - {path}"
                )
            aoi_map[stem] = path
    if not aoi_map:
        raise ValueError(
            "No AOI files matched aoi_vector_pattern. To intentionally analyze "
            "the full overlapping raster extent, set "
            "inputs.analyze_full_raster_extent: true and leave "
            "aoi_vector_pattern empty."
        )
    return aoi_map


def _chunks(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])


def _sum_raster_blocks(raster_path):
    total = np.float64(0)
    for _, block in geoprocessing.iterblocks(
        (str(raster_path), 1),
        largest_block=10 * 2**20,
    ):
        total += np.sum(block, dtype=np.float64)
    return total


def subset_subwatersheds(
    aoi_vector_path: str | Path,
    subwatershed_vector_path: str | Path,
    target_crs: PickedCRS,
    target_subset_subwatersheds_vector_path: str | Path,
) -> None:
    """Extract and save all sub-watersheds downstream of a given AOI.

    This function reads an area of interest (AOI) polygon, identifies the set
    of sub-watersheds in the input vector layer that intersect the AOI, and
    traverses their `NEXT_DOWN` links to collect all downstream sub-watersheds.
    The resulting geometries are re-projected to the provided target CRS and
    written to disk.

    Args:
        aoi_vector_path (str | Path): Path to the AOI vector dataset
            (any format readable by GeoPandas/pyogrio).
        subwatershed_vector_path (str | Path): Path to the sub-watershed
            dataset containing `HYBAS_ID`, `NEXT_DOWN`, and geometry columns.
        target_crs (PickedCRS): Target coordinate reference system for the
            output, typically selected by a helper such as
            ``choose_equidistant_crs_from_bbox``.
        target_subset_subwatersheds_vector_path (str | Path): Destination
            file path where the subset of downstream sub-watersheds will be
            written (e.g., a GeoPackage ``.gpkg``) and CRS is explicitly set to
            ``target_crs``

    Returns:
        None: The function writes the downstream sub-watersheds to the specified
        file and does not return anything.

    Raises:
        ValueError: If required attributes (``HYBAS_ID`` or ``NEXT_DOWN``) are
            missing, or if the AOI and sub-watershed geometries do not overlap.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"processing aoi {aoi_vector_path}")

    # Read AOI and precompute union + bbox; repair invalid with make_valid if available
    aoi_gdf = gpd.read_file(aoi_vector_path)
    aoi_crs = aoi_gdf.crs
    aoi_gdf = aoi_gdf.set_geometry(aoi_gdf.geometry.make_valid())
    aoi_union = aoi_gdf.geometry.union_all()
    aoi_bbox = box(*aoi_gdf.total_bounds)

    logger.debug(f"about to read info for {subwatershed_vector_path}")
    subwatershed_info = pyogrio.read_info(subwatershed_vector_path)
    logger.debug(f"successfully read info for {subwatershed_vector_path}")
    sub_crs = (
        CRS.from_user_input(subwatershed_info.get("crs"))
        if subwatershed_info.get("crs")
        else aoi_crs
    )

    if aoi_crs != sub_crs:
        tform = Transformer.from_crs(aoi_crs, sub_crs, always_xy=True).transform
        aoi_union = transform(tform, aoi_union)
        aoi_bbox = transform(tform, aoi_bbox)

    attrs = pyogrio.read_dataframe(
        subwatershed_vector_path,
        columns=["HYBAS_ID", "NEXT_DOWN"],
        read_geometry=False,
    )
    hybas_to_nextdown = dict(
        zip(attrs["HYBAS_ID"].to_numpy(), attrs["NEXT_DOWN"].to_numpy())
    )

    logger.debug(f"spatial pre-filter {aoi_vector_path} w/ bb")
    sub_bbox_gdf = pyogrio.read_dataframe(
        subwatershed_vector_path,
        bbox=aoi_bbox.bounds,
        columns=["HYBAS_ID", "NEXT_DOWN", "geometry"],
    )
    if sub_bbox_gdf.empty:
        raise ValueError(f"No candidates found in bbox for {aoi_vector_path}.")

    hits = sub_bbox_gdf.sindex.query(aoi_union, predicate="intersects")
    if len(hits) == 0:
        raise ValueError(f"No intersecting sub-watersheds for {aoi_vector_path}.")

    initial = sub_bbox_gdf.iloc[hits]
    initial_ids = set(initial["HYBAS_ID"].tolist())
    visited_ids = set(initial_ids)
    ds_ids_to_process = set(initial["NEXT_DOWN"].tolist())
    ds_ids_to_process.discard(0)  # 0 is the outlet

    while ds_ids_to_process:
        visited_ids.update(ds_ids_to_process)
        next_ids = {
            hybas_to_nextdown.get(h)
            for h in ds_ids_to_process
            if h in hybas_to_nextdown
        }
        next_ids.discard(None)
        next_ids.discard(0)
        ds_ids_to_process = next_ids - visited_ids

    if not visited_ids:
        raise ValueError(f"No valid geometry found for {aoi_vector_path}.")

    # Fetch geometries for all_ids using batched attribute filters
    logger.debug(f"fetch geometries by attribute filter {aoi_vector_path}")
    downstream_features = []
    for id_chunk in _chunks(sorted(visited_ids), 1000):
        where = f'HYBAS_ID IN ({",".join(map(str, id_chunk))})'
        df = pyogrio.read_dataframe(
            subwatershed_vector_path,
            where=where,
            columns=None,
        )
        downstream_features.append(df)

    if not downstream_features:
        raise ValueError(f"No geometries returned for {aoi_vector_path}.")

    sub_gdf = pd.concat(downstream_features, ignore_index=True)
    sub_gdf = gpd.GeoDataFrame(sub_gdf, crs=sub_crs)

    # Repair invalid geometries if needed
    invalid = ~sub_gdf.geometry.is_valid
    if invalid.any():
        if hasattr(sub_gdf.geometry, "make_valid"):
            sub_gdf.loc[invalid, "geometry"] = sub_gdf.loc[
                invalid, "geometry"
            ].make_valid()
        else:
            sub_gdf.loc[invalid, "geometry"] = sub_gdf.loc[invalid, "geometry"].buffer(
                0
            )

    # Reproject to AOI CRS and write
    if sub_crs and aoi_crs and sub_crs != aoi_crs:
        sub_gdf = sub_gdf.to_crs(aoi_crs)

    sub_gdf = sub_gdf.to_crs(target_crs.crs)
    sub_gdf.to_file(target_subset_subwatersheds_vector_path, driver="GPKG")
    logger.debug(f"all done subwatershedding {aoi_vector_path}")


def partition_subwatersheds_by_terminal_drain(
    aoi_vector_path: str | Path,
    subwatershed_vector_path: str | Path,
    target_crs: PickedCRS,
    target_partition_dir: str | Path,
    debug_drain_index: int | None = None,
) -> dict[str, Path]:
    """Write downstream subwatershed partitions grouped by terminal drain.

    Args:
        aoi_vector_path: Path to the AOI vector dataset.
        subwatershed_vector_path: Path to a HydroBASINS-style vector dataset
            containing ``HYBAS_ID``, ``NEXT_DOWN``, and ``NEXT_SINK`` columns.
        target_crs: CRS to use for the written partition vectors.
        target_partition_dir: Directory where partition GeoPackages should be
            written.
        debug_drain_index: Optional zero-based index into the sorted terminal
            drain partitions. If set, only that partition is read and written.

    Returns:
        Mapping from partition id to the written partition vector path. Each
        partition contains downstream subwatersheds with the same
        ``NEXT_SINK`` id.

    Raises:
        ValueError: If the AOI does not intersect subwatersheds, the
            downstream graph is incomplete, or ``debug_drain_index`` is outside
            the available partition range.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"partitioning downstream subwatersheds from {aoi_vector_path}")

    aoi_gdf = gpd.read_file(aoi_vector_path)
    aoi_crs = aoi_gdf.crs
    aoi_gdf = aoi_gdf.set_geometry(aoi_gdf.geometry.make_valid())
    aoi_union = aoi_gdf.geometry.union_all()
    aoi_bbox = box(*aoi_gdf.total_bounds)

    subwatershed_info = pyogrio.read_info(subwatershed_vector_path)
    sub_crs = (
        CRS.from_user_input(subwatershed_info.get("crs"))
        if subwatershed_info.get("crs")
        else aoi_crs
    )

    if aoi_crs != sub_crs:
        tform = Transformer.from_crs(aoi_crs, sub_crs, always_xy=True).transform
        aoi_union = transform(tform, aoi_union)
        aoi_bbox = transform(tform, aoi_bbox)

    with tqdm(
        total=1,
        desc="read watershed topology",
        unit="table",
        dynamic_ncols=True,
        leave=False,
    ) as progress:
        attrs = pyogrio.read_dataframe(
            subwatershed_vector_path,
            columns=["HYBAS_ID", "NEXT_DOWN", "NEXT_SINK"],
            read_geometry=False,
        )
        progress.update()
    hybas_to_nextdown = dict(
        zip(attrs["HYBAS_ID"].to_numpy(), attrs["NEXT_DOWN"].to_numpy())
    )
    hybas_to_nextsink = dict(
        zip(attrs["HYBAS_ID"].to_numpy(), attrs["NEXT_SINK"].to_numpy())
    )

    with tqdm(
        total=1,
        desc="read AOI candidate watersheds",
        unit="table",
        dynamic_ncols=True,
        leave=False,
    ) as progress:
        sub_bbox_gdf = pyogrio.read_dataframe(
            subwatershed_vector_path,
            bbox=aoi_bbox.bounds,
            columns=["HYBAS_ID", "NEXT_DOWN", "NEXT_SINK", "geometry"],
        )
        progress.update()
    if sub_bbox_gdf.empty:
        raise ValueError(f"No candidates found in bbox for {aoi_vector_path}.")

    hits = sub_bbox_gdf.sindex.query(aoi_union, predicate="intersects")
    if len(hits) == 0:
        raise ValueError(f"No intersecting sub-watersheds for {aoi_vector_path}.")

    initial = sub_bbox_gdf.iloc[hits]
    visited_ids = set(initial["HYBAS_ID"].tolist())
    ds_ids_to_process = set(initial["NEXT_DOWN"].tolist())
    ds_ids_to_process.discard(0)

    with tqdm(
        desc="collect downstream watersheds",
        unit="watershed",
        dynamic_ncols=True,
        leave=False,
    ) as progress:
        progress.update(len(visited_ids))
        while ds_ids_to_process:
            new_downstream_ids = ds_ids_to_process - visited_ids
            progress.set_postfix(
                queued=len(ds_ids_to_process),
                discovered=len(visited_ids) + len(new_downstream_ids),
                refresh=False,
            )
            visited_ids.update(ds_ids_to_process)
            progress.update(len(new_downstream_ids))
            next_ids = {
                hybas_to_nextdown.get(h)
                for h in ds_ids_to_process
                if h in hybas_to_nextdown
            }
            next_ids.discard(None)
            next_ids.discard(0)
            ds_ids_to_process = next_ids - visited_ids

    if not visited_ids:
        raise ValueError(f"No valid geometry found for {aoi_vector_path}.")

    ids_by_next_sink = collections.defaultdict(set)
    for hybas_id in tqdm(
        visited_ids,
        desc="group watersheds by sink",
        unit="watershed",
        dynamic_ncols=True,
        leave=False,
    ):
        next_sink_id = hybas_to_nextsink.get(hybas_id)
        if next_sink_id is None or pd.isna(next_sink_id):
            raise ValueError(f"Could not find NEXT_SINK for HYBAS_ID {hybas_id}.")
        ids_by_next_sink[next_sink_id].add(hybas_id)

    target_partition_dir = Path(target_partition_dir)
    target_partition_dir.mkdir(parents=True, exist_ok=True)

    partition_items = sorted(ids_by_next_sink.items())
    full_partition_count = len(partition_items)
    if debug_drain_index is not None:
        if not partition_items:
            raise ValueError(
                "`inputs.debug_drain_index` was set, but no drain partitions "
                "are available."
            )
        if debug_drain_index >= full_partition_count:
            raise ValueError(
                "`inputs.debug_drain_index` is out of range: "
                f"{debug_drain_index}. Available drain partition index range is "
                f"0-{full_partition_count - 1}."
            )
        partition_items = [partition_items[debug_drain_index]]
        selected_partition_id = f"drain_{partition_items[0][0]}"
        logger.info(
            "debug_drain_index=%d selected %s; writing 1 of %d drain partitions",
            debug_drain_index,
            selected_partition_id,
            full_partition_count,
        )

    ids_to_read = set()
    for _, partition_ids in partition_items:
        ids_to_read.update(partition_ids)

    downstream_features = []
    chunk_size = 1000
    total_chunks = math.ceil(len(ids_to_read) / chunk_size)
    for id_chunk in tqdm(
        _chunks(sorted(ids_to_read), chunk_size),
        desc="read downstream watershed geometries",
        total=total_chunks,
        unit="chunk",
        dynamic_ncols=True,
        leave=False,
    ):
        where = f'HYBAS_ID IN ({",".join(map(str, id_chunk))})'
        df = pyogrio.read_dataframe(
            subwatershed_vector_path,
            where=where,
            columns=None,
        )
        downstream_features.append(df)

    if not downstream_features:
        raise ValueError(f"No geometries returned for {aoi_vector_path}.")

    sub_gdf = pd.concat(downstream_features, ignore_index=True)
    sub_gdf = gpd.GeoDataFrame(sub_gdf, crs=sub_crs)
    sub_gdf = sub_gdf.drop_duplicates(subset=["HYBAS_ID"])

    invalid = ~sub_gdf.geometry.is_valid
    if invalid.any():
        if hasattr(sub_gdf.geometry, "make_valid"):
            sub_gdf.loc[invalid, "geometry"] = sub_gdf.loc[
                invalid, "geometry"
            ].make_valid()
        else:
            sub_gdf.loc[invalid, "geometry"] = sub_gdf.loc[invalid, "geometry"].buffer(
                0
            )

    if sub_crs and aoi_crs and sub_crs != aoi_crs:
        sub_gdf = sub_gdf.to_crs(aoi_crs)
    sub_gdf = sub_gdf.to_crs(target_crs.crs)

    partition_paths = {}
    for next_sink_id, partition_ids in tqdm(
        partition_items,
        desc="write drain partitions",
        unit="partition",
        dynamic_ncols=True,
    ):
        partition_id = f"drain_{next_sink_id}"
        partition_path = target_partition_dir / f"{partition_id}.gpkg"
        partition_gdf = sub_gdf[sub_gdf["HYBAS_ID"].isin(partition_ids)].copy()
        if partition_gdf.empty:
            raise ValueError(f"No geometries found for NEXT_SINK {next_sink_id}.")
        partition_gdf.to_file(partition_path, driver="GPKG")
        partition_paths[partition_id] = partition_path

    logger.info(
        "wrote %d drain partitions for %s",
        len(partition_paths),
        aoi_vector_path,
    )
    return partition_paths


def align_and_resize_raster_on_vector(
    raster_path,
    target_path,
    resample_method,
    target_pixel_size,
    bounding_vector_path,
):
    gdf = gpd.read_file(bounding_vector_path)
    gdf = gdf.set_geometry(gdf.geometry.make_valid())
    if not gdf.crs or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    target_bb = [float(x) for x in gdf.total_bounds]
    target_projection_wkt = CRS.from_user_input(gdf.crs).to_wkt()
    geoprocessing.warp_raster(
        raster_path,
        target_pixel_size,
        target_path,
        resample_method,
        target_bb=target_bb,
        target_projection_wkt=target_projection_wkt,
        working_dir=Path(target_path).parent,
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )
    mask_raster_to_vector(target_path, bounding_vector_path)


def mask_raster_to_vector(raster_path: str | Path, vector_path: str | Path) -> None:
    """Mask a raster in place to a vector geometry."""
    vector_gdf = gpd.read_file(vector_path)
    vector_gdf = vector_gdf.set_geometry(vector_gdf.geometry.make_valid())
    if vector_gdf.empty:
        raise ValueError(f"No geometries found in {vector_path}.")

    with rasterio.open(raster_path, "r+") as raster:
        if raster.crs is None:
            raise ValueError(f"Raster has no CRS and cannot be masked: {raster_path}")
        if vector_gdf.crs is None:
            raise ValueError(f"Vector has no CRS and cannot mask raster: {vector_path}")
        if vector_gdf.crs != raster.crs:
            vector_gdf = vector_gdf.to_crs(raster.crs)

        geometries = [
            geom
            for geom in vector_gdf.geometry
            if geom is not None and not geom.is_empty
        ]
        if not geometries:
            raise ValueError(f"No valid geometries found in {vector_path}.")

        outside_value = raster.nodata if raster.nodata is not None else 0
        for _, window in raster.block_windows(1):
            block = raster.read(1, window=window)
            inside_mask = rasterio.features.geometry_mask(
                geometries,
                out_shape=(int(window.height), int(window.width)),
                transform=rasterio.windows.transform(window, raster.transform),
                invert=True,
            )
            block[~inside_mask] = outside_value
            raster.write(block, 1, window=window)


def eck4_limits(r=6371000):
    crs_eck4 = CRS.from_proj4(f"+proj=eck4 +R={r} +units=m +no_defs")
    T_fwd = Transformer.from_crs("EPSG:4326", crs_eck4, always_xy=True)
    xs, ys = T_fwd.transform([-180, 180, 0, 0], [0, 0, 90, -90])
    return abs(xs[0]), abs(ys[2])  # xmax, ymax


ECKERT_IV_MAX_X, ECKERT_IV_MAX_Y = eck4_limits()  # ≈ 15 110 000 , 7 540 000


def _is_eckert_iv_crs(crs):
    """Return True when ``crs`` uses an Eckert IV projection."""
    if crs is None:
        return False
    coordinate_operation = CRS.from_user_input(crs).coordinate_operation
    if coordinate_operation is None:
        return False
    return coordinate_operation.method_name == "Eckert IV"


def _clamp_eckert_point(x, y):
    r2 = (x * x) / (ECKERT_IV_MAX_X * ECKERT_IV_MAX_X) + (y * y) / (
        ECKERT_IV_MAX_Y * ECKERT_IV_MAX_Y
    )
    if r2 <= 1:
        return x, y
    scale = 0.999 / r2**2
    return x * scale, y * scale


def transform_edge_points_eckert_to_wgs84(bbox_gdf, dst_crs="EPSG:4326"):
    if bbox_gdf.crs is None:
        raise ValueError("bbox_gdf must have a CRS defined")
    if not _is_eckert_iv_crs(bbox_gdf.crs):
        return bbox_gdf.to_crs(dst_crs)

    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    raw_corners = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
    safe_corners = [_clamp_eckert_point(x, y) for x, y in raw_corners]

    transformer = Transformer.from_crs(bbox_gdf.crs, dst_crs, always_xy=True)
    lonlat = [transformer.transform(x, y) for x, y in safe_corners]
    xs, ys = zip(*lonlat)
    proj_box = box(min(xs), min(ys), max(xs), max(ys))
    return gpd.GeoDataFrame(geometry=[proj_box], crs=dst_crs)


def _clip_and_reproject_raster(
    base_raster_path, bbox_gdf, dst_crs, target_raster_path, reference_meta=None
):
    logger = logging.getLogger(__name__)
    with rasterio.open(base_raster_path) as src:
        if bbox_gdf.crs is None:
            raise ValueError("bbox_gdf must have a CRS defined")

        if _is_eckert_iv_crs(bbox_gdf.crs):
            # eckert is so broken, just doing regular lat/lng bounds
            projected_box_gdf = gpd.GeoDataFrame(
                geometry=[box(-179, -80, 179, 80)], crs="EPSG:4326"
            )
        else:
            # Check intermediate clamped bounds explicitly:
            logger.debug(f"Clamped bbox bounds: {bbox_gdf.total_bounds}")

            # Safely project to src.crs (e.g., EPSG:4326)
            projected_box_gdf = bbox_gdf.to_crs(src.crs)

        bbox_projected_geom = [projected_box_gdf.geometry.iloc[0]]
        out_image, out_transform = rasterio.mask.mask(
            src, bbox_projected_geom, crop=True
        )

        if reference_meta:
            dst_transform = reference_meta["transform"]
            width = reference_meta["width"]
            height = reference_meta["height"]
        else:
            dst_transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                out_image.shape[2],
                out_image.shape[1],
                *projected_box_gdf.total_bounds,
            )

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
            }
        )
        _set_tiled_geotiff_creation_options(out_meta)

        reproject_kwargs = {}
        if src.nodata is not None:
            reproject_kwargs["src_nodata"] = src.nodata
            reproject_kwargs["dst_nodata"] = src.nodata

        with rasterio.open(target_raster_path, "w", **out_meta) as dst:
            reproject(
                source=out_image,
                destination=rasterio.band(dst, 1),
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                **reproject_kwargs,
            )


def _buffer_window(window: Window, buffer_pixels: int, width: int, height: int) -> Window:
    """Expand ``window`` by ``buffer_pixels`` and clamp to raster dimensions."""
    return _integer_window(
        Window(
            window.col_off - buffer_pixels,
            window.row_off - buffer_pixels,
            window.width + 2 * buffer_pixels,
            window.height + 2 * buffer_pixels,
        ),
        width,
        height,
    )


def _estimate_travel_reach_bytes(width: int, height: int, friction_dtype) -> int:
    """Estimate array memory needed by one ``find_mask_reach`` window."""
    pixel_count = int(width) * int(height)
    friction_bytes = np.dtype(friction_dtype).itemsize
    return pixel_count * (friction_bytes + np.dtype(np.int8).itemsize + 5)


def _raster_block_count(width: int, height: int, block_size: int) -> int:
    """Return the number of block windows needed to cover a raster."""
    return math.ceil(width / block_size) * math.ceil(height / block_size)


def _write_mask_raster_by_window(
    target_mask_raster_path,
    raster_profile,
    source_geometries,
    use_wgs84_bounds_mask=False,
    wgs84_bounds=None,
):
    """Write a binary AOI mask raster without allocating the full raster."""
    target_mask_raster_path = Path(target_mask_raster_path)
    target_mask_raster_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target_mask_raster_path, "w", **raster_profile) as target:
        block_count = _raster_block_count(
            target.width,
            target.height,
            RASTER_BLOCK_SIZE,
        )
        for _, window in tqdm(
            target.block_windows(1),
            total=block_count,
            desc=f"rasterize {target_mask_raster_path.stem}",
            unit="block",
        ):
            if use_wgs84_bounds_mask:
                mask_array = _rasterize_wgs84_bounds_mask(
                    wgs84_bounds,
                    rasterio.windows.transform(window, target.transform),
                    target.crs,
                    int(window.height),
                    int(window.width),
                ).astype(np.uint8, copy=False)
            else:
                mask_array = rasterio.features.rasterize(
                    ((geom, 1) for geom in source_geometries),
                    out_shape=(int(window.height), int(window.width)),
                    transform=rasterio.windows.transform(window, target.transform),
                    fill=0,
                    dtype=rasterio.uint8,
                )
            target.write(mask_array, 1, window=window)


def calculate_windowed_travel_reach(
    friction_raster_path,
    source_mask_raster_path,
    target_coverage_raster_path,
    max_time_mins,
    buffer_pixels,
    core_block_size=RASTER_BLOCK_SIZE,
    max_window_bytes=TRAVEL_TIME_MAX_WINDOW_BYTES,
    progress_label=None,
):
    """Calculate travel reach by OR-ing buffered source-mask windows.

    The reach operation is equivalent to running the whole source mask at
    once, but each Dijkstra expansion is bounded to a core source block plus
    the maximum possible travel distance. This keeps memory proportional to a
    buffered block instead of to the full drain basin.
    """
    logger = logging.getLogger(__name__)
    target_coverage_raster_path = Path(target_coverage_raster_path)

    with rasterio.open(friction_raster_path) as friction:
        target_profile = friction.profile.copy()
        target_profile.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0})
        _set_tiled_geotiff_creation_options(target_profile)
        cell_length_m = abs(friction.transform.a)
        full_window = Window(0, 0, friction.width, friction.height)
        total_windows = _raster_block_count(
            friction.width,
            friction.height,
            core_block_size,
        )
        logger.info(
            "windowed travel reach: raster=%sx%s px, core=%s px, "
            "buffer=%s px, max_time=%.1f min",
            friction.width,
            friction.height,
            core_block_size,
            buffer_pixels,
            max_time_mins,
        )

    _create_zeroed_raster(
        target_coverage_raster_path,
        target_profile,
        desc=f"initialize {target_coverage_raster_path.stem}",
    )

    processed_windows = 0
    skipped_windows = 0
    with rasterio.open(friction_raster_path) as friction:
        with rasterio.open(source_mask_raster_path) as source_mask:
            with rasterio.open(target_coverage_raster_path, "r+") as target:
                progress = tqdm(
                    _iter_block_windows(full_window, core_block_size),
                    total=total_windows,
                    desc=progress_label or f"travel windows {target_coverage_raster_path.stem}",
                    unit="window",
                )
                for window_index, core_window in enumerate(progress, start=1):
                    core_mask = source_mask.read(1, window=core_window)
                    if not np.any(core_mask == 1):
                        skipped_windows += 1
                        continue

                    buffered_window = _buffer_window(
                        core_window,
                        buffer_pixels,
                        friction.width,
                        friction.height,
                    )
                    estimated_bytes = _estimate_travel_reach_bytes(
                        int(buffered_window.width),
                        int(buffered_window.height),
                        friction.dtypes[0],
                    )
                    if estimated_bytes > max_window_bytes:
                        raise ValueError(
                            "Travel-time reach window is too large: "
                            f"{int(buffered_window.width)}x"
                            f"{int(buffered_window.height)} px, estimated "
                            f"{estimated_bytes / 1024**3:.2f} GiB. "
                            "Reduce the travel-time core block size or review "
                            "the travel-time buffer distance."
                        )

                    friction_array = friction.read(1, window=buffered_window).astype(
                        np.float32,
                        copy=False,
                    )
                    source_array = np.zeros(friction_array.shape, dtype=np.int8)
                    row_start = int(core_window.row_off - buffered_window.row_off)
                    col_start = int(core_window.col_off - buffered_window.col_off)
                    row_stop = row_start + int(core_window.height)
                    col_stop = col_start + int(core_window.width)
                    source_array[row_start:row_stop, col_start:col_stop] = (
                        core_mask == 1
                    ).astype(np.int8)

                    n_rows, n_cols = friction_array.shape
                    reach_label = (
                        "find_mask_reach "
                        f"{target_coverage_raster_path.stem} "
                        f"window={window_index}/{total_windows} "
                        f"core={int(core_window.width)}x{int(core_window.height)} "
                        f"buffered={n_cols}x{n_rows} "
                        f"estimated_arrays={_format_bytes(estimated_bytes)}"
                    )
                    with _log_memory_scope(logger, reach_label):
                        reach_array = shortest_distances.find_mask_reach(
                            friction_array,
                            source_array,
                            cell_length_m,
                            n_cols,
                            n_rows,
                            max_time_mins,
                            progress_interval_seconds=0,
                        )

                    existing_array = target.read(1, window=buffered_window)
                    np.maximum(existing_array, reach_array, out=existing_array)
                    target.write(existing_array, 1, window=buffered_window)
                    processed_windows += 1
                    progress.set_postfix(
                        processed=processed_windows,
                        skipped=skipped_windows,
                    )

    logger.info(
        "wrote windowed travel coverage %s from %d source windows",
        target_coverage_raster_path,
        processed_windows,
    )
    return target_coverage_raster_path


def calculate_travel_time_coverage(
    traveltime_raster_path: Path,
    aoi_vector_path: Path,
    max_hours: float,
    travel_time_pixel_size_m: float,
    target_coverage_raster_path: Path,
    working_dir: Path,
    use_wgs84_bounds_mask: bool = False,
):
    """Create a binary travel-time coverage raster clipped to an AOI.

    Args:
        traveltime_raster_path (str | Path): Path to a raster where each pixel
            encodes travel time (hours) to a target destination.
        aoi_vector_path (str | Path): Path to a vector dataset (e.g. Shapefile,
            GeoPackage) defining the area of interest to clip to.
        max_hours (float): Maximum travel time in hours to include.
        travel_time_pixel_size_m (float): Target travel-time raster pixel size
            in meters.
        use_wgs84_bounds_mask (bool): If true, rasterize the AOI as a WGS84
            bounds mask from pixel centers. This is intended for generated
            full-extent AOIs, where projecting a near-global rectangle into a
            local projected CRS can collapse the rasterized mask.

    Returns:
        Path: ``target_coverage_raster_path``.
    """
    configure_gdal_cache()
    logger = logging.getLogger(__name__)
    worker_label = f"travel-time coverage worker {Path(aoi_vector_path).stem}"
    worker_start_time = time.monotonic()
    worker_start_rss = _process_tree_rss_bytes()
    logger.info("%s started; %s", worker_label, _memory_status_text())
    max_time_mins = max_hours * 60
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    analysis_crs = choose_equidistant_crs_from_bbox(str(aoi_vector_path)).crs
    aoi_vector = gpd.read_file(aoi_vector_path)
    aoi_wgs84_bounds = aoi_vector.to_crs("EPSG:4326").total_bounds
    projected_gdf = aoi_vector.to_crs(analysis_crs)
    bbox = projected_gdf.total_bounds
    # This max-distance bound is precomputed for the global friction raster
    # used by this workflow.
    buffer_distance_m = max_hours * TRAVEL_TIME_MAX_DISTANCE_M_PER_HOUR

    buffered_bbox = box(
        bbox[0] - buffer_distance_m,
        bbox[1] - buffer_distance_m,
        bbox[2] + buffer_distance_m,
        bbox[3] + buffer_distance_m,
    )

    logger.debug(f"buffered box: {buffered_bbox}")
    bbox_gdf = gpd.GeoDataFrame({"geometry": [buffered_bbox]}, crs=projected_gdf.crs)

    target_friction_clipped_raster_path = Path(
        working_dir / f"{traveltime_raster_path.stem}_friction_clip.tif"
    )
    target_aoi_raster_path = Path(working_dir / "travel_time_aoi_mask.tif")

    logger.info(
        "preparing travel-time coverage for %s in %s",
        Path(aoi_vector_path).stem,
        analysis_crs,
    )
    with _log_heartbeat(
        logger,
        lambda: (
            "warping travel-time raster for "
            f"{Path(aoi_vector_path).stem} to {target_friction_clipped_raster_path}"
        ),
    ):
        geoprocessing.warp_raster(
            str(traveltime_raster_path),
            (travel_time_pixel_size_m, -travel_time_pixel_size_m),
            str(target_friction_clipped_raster_path),
            "near",
            target_bb=buffered_bbox.bounds,
            target_projection_wkt=analysis_crs.to_wkt(),
            working_dir=str(working_dir),
            output_type=gdal.GDT_Float32,
            raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
        )

    with rasterio.open(target_friction_clipped_raster_path) as friction_ref:
        ref_meta = friction_ref.profile.copy()
        cell_length_m = abs(friction_ref.transform.a)
        buffer_pixels = int(math.ceil(buffer_distance_m / cell_length_m))
        logger.info(
            "travel-time window setup for %s: raster=%sx%s px, "
            "buffer=%s m (%s px)",
            Path(aoi_vector_path).stem,
            friction_ref.width,
            friction_ref.height,
            int(buffer_distance_m),
            buffer_pixels,
        )

    aoi_meta = ref_meta.copy()
    aoi_meta.update(
        {"count": 1, "dtype": rasterio.uint8, "nodata": 0, "compress": "lzw"}
    )
    _set_tiled_geotiff_creation_options(aoi_meta)

    _write_mask_raster_by_window(
        target_aoi_raster_path,
        aoi_meta,
        projected_gdf.geometry,
        use_wgs84_bounds_mask=use_wgs84_bounds_mask,
        wgs84_bounds=aoi_wgs84_bounds,
    )

    result_path = calculate_windowed_travel_reach(
        target_friction_clipped_raster_path,
        target_aoi_raster_path,
        target_coverage_raster_path,
        max_time_mins,
        buffer_pixels,
        progress_label=f"travel windows {Path(aoi_vector_path).stem}",
    )
    logger.info(
        "%s finished in %.1fs; rss_delta=%s; %s",
        worker_label,
        time.monotonic() - worker_start_time,
        _format_bytes(_process_tree_rss_bytes() - worker_start_rss),
        _memory_status_text(),
    )
    return result_path


def create_distance_transform(
    base_mask_raster_path,
    target_distance_transform_path,
    target_nodata=DISTANCE_TRANSFORM_NODATA,
):
    """Create a distance-transform raster from a binary mask.

    This function computes a distance transform on a binary mask raster using
    ``gdal.ComputeProximity``. Distances are computed in pixel units from all
    pixels whose value is exactly ``1`` in the input raster. The resulting
    distance raster is written as a tiled, compressed GeoTIFF that inherits
    the geotransform and projection from the source mask.

    Args:
        base_mask_raster_path: Path-like to a single-band raster containing a
            binary mask. Pixels with value ``1`` are treated as source pixels
            for the distance transform; all other values are treated as
            non-source.
        target_distance_transform_path: Path-like specifying where the
            output distance-transform GeoTIFF should be written.
        target_nodata: Nodata value to assign to pixels that GDAL proximity
            does not compute.

    Returns:
        None
    """
    base_raster = gdal.Open(base_mask_raster_path)
    src_band = base_raster.GetRasterBand(1)

    driver = gdal.GetDriverByName("GTiff")
    target_raster = driver.Create(
        target_distance_transform_path,
        base_raster.RasterXSize,
        base_raster.RasterYSize,
        1,
        gdal.GDT_Float32,
        GTIFF_CREATION_OPTIONS,
    )

    target_raster.SetGeoTransform(base_raster.GetGeoTransform())
    target_raster.SetProjection(base_raster.GetProjection())
    target_band = target_raster.GetRasterBand(1)
    target_band.SetNoDataValue(target_nodata)
    target_band.Fill(target_nodata)

    gdal.ComputeProximity(
        src_band,
        target_band,
        ["VALUES=1", "DISTUNITS=PIXEL", f"NODATA={target_nodata}"],
    )

    target_raster = None
    base_raster = None


def make_condition_mask_op(base_raster_nodata, expression):
    """Build a raster calculator op for a binary condition mask.

    The returned function evaluates ``expression`` only on valid source pixels.
    Source nodata pixels are always false so they cannot leak into the
    downstream source mask as nonzero byte values.
    """

    def _condition_mask_op(value):
        result = np.zeros(value.shape, dtype=np.uint8)
        valid_mask = (
            np.ones(value.shape, dtype=bool)
            if base_raster_nodata is None
            else value != base_raster_nodata
        )
        if not np.any(valid_mask):
            return result

        expression_result = eval(
            expression,
            {"__builtins__": {}},
            {"value": value[valid_mask], "np": np},
        )
        result[valid_mask] = np.asarray(expression_result).astype(bool).astype(np.uint8)
        return result

    return _condition_mask_op


def calculate_downstream_coverage_from_conditional_raster(
    aoi_vector_path,
    flow_dir_raster_path,
    base_raster_path,
    condition_id,
    expression,
    buffer_size_m,
    max_downstream_distance_m,
    travel_time_pixel_size_m,
    working_dir,
    target_coverage_raster_path,
):
    """Calculate downstream coverage satisfying a conditional raster expression.

    This function evaluates a user-provided expression on a base raster to
    create a binary condition mask, routes that mask downstream using
    multiple-flow-direction accumulation, buffers it, and optionally truncates
    it by a maximum downstream distance. The final output is a binary coverage
    raster.

    Intermediate rasters (condition mask, clipped base, downstream coverage,
    buffer kernel, distance transform, and optionally distance-limited
    coverage) are created in ``working_dir`` with filenames derived from
    ``condition_id`` and ``base_raster_path``.

    Args:
        aoi_vector_path: Path-like to a vector dataset defining the analysis
            area of interest. Used to mask the warped base raster.
        flow_dir_raster_path: Path-like to a flow-direction raster used as
            input to ``routing.flow_accumulation_mfd``.
        base_raster_path: Path-like to the base raster on which
            ``expression`` is evaluated to derive the condition mask.
        condition_id: Identifier (string or value convertible to string) used
            to differentiate intermediate filenames for this condition.
        expression: String containing a Python expression evaluated with
            ``value`` (a NumPy array of base raster values) and ``np``
            (NumPy). Nonzero results define the condition mask.
        buffer_size_m: Circular buffer radius in meters used to convolve the
            downstream coverage raster.
        max_downstream_distance_m: Optional maximum downstream distance in
            meters. If not ``None``, a distance transform is used to zero out
            coverage beyond this distance.
        travel_time_pixel_size_m: Pixel size in meters used to convert the
            distance-transform pixel counts to meters when applying
            ``max_downstream_distance_m``.
        working_dir: Path-like directory where all intermediate rasters and
            kernels will be written.
        target_coverage_raster_path: Path-like where the final binary coverage
            raster will be written.

    Returns:
        Path: ``target_coverage_raster_path``.
    """
    configure_gdal_cache()
    logger = logging.getLogger(__name__)
    worker_label = f"downstream coverage worker {condition_id} {Path(aoi_vector_path).stem}"
    worker_start_time = time.monotonic()
    worker_start_rss = _process_tree_rss_bytes()
    logger.info("%s started; %s", worker_label, _memory_status_text())
    logger.debug(f"max downstream distance: {max_downstream_distance_m}")
    condition_raster_path = working_dir / f"mask_{condition_id}_{base_raster_path.name}"

    clipped_base_raster_path = (
        working_dir / f"clipped_{condition_id}_{base_raster_path.name}"
    )

    flow_dir_info = geoprocessing.get_raster_info(flow_dir_raster_path)
    base_info = geoprocessing.get_raster_info(base_raster_path)
    geoprocessing.warp_raster(
        base_raster_path,
        flow_dir_info["pixel_size"],
        clipped_base_raster_path,
        "near",
        target_bb=flow_dir_info["bounding_box"],
        target_projection_wkt=flow_dir_info["projection_wkt"],
        working_dir=working_dir,
        output_type=base_info["datatype"],
        vector_mask_options={
            "mask_vector_path": aoi_vector_path,
        },
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )

    base_raster_nodata = base_info["nodata"][0]

    geoprocessing.raster_calculator(
        [(str(clipped_base_raster_path), 1)],
        make_condition_mask_op(base_raster_nodata, expression),
        condition_raster_path,
        gdal.GDT_Byte,
        0,
        calc_raster_stats=False,
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )

    ds_coverage_raster_path = str(
        working_dir / f"ds_coverage_{condition_id}_{base_raster_path.name}"
    )

    routing.flow_accumulation_mfd(
        (str(flow_dir_raster_path), 1),
        str(ds_coverage_raster_path),
        weight_raster_path_band=(str(condition_raster_path), 1),
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )

    buffer_amounts_in_pixels = int(np.round(buffer_size_m / travel_time_pixel_size_m))

    kernel_path = str(
        working_dir / f"{buffer_amounts_in_pixels}_{condition_id}_kernel.tif"
    )
    create_circular_kernel(kernel_path, buffer_amounts_in_pixels)
    buffered_ds_coverage_raster_path = "%s_buff%s" % os.path.splitext(
        str(ds_coverage_raster_path)
    )
    geoprocessing.convolve_2d(
        (ds_coverage_raster_path, 1),
        (kernel_path, 1),
        buffered_ds_coverage_raster_path,
        n_workers=1,
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )
    if max_downstream_distance_m is not None:

        def _distance_mask_op(mask, n_pixels):
            return (
                (n_pixels != DISTANCE_TRANSFORM_NODATA)
                & (mask > DOWNSTREAM_COVERAGE_EPSILON)
                & (n_pixels * travel_time_pixel_size_m <= max_downstream_distance_m)
            )

        distance_transform_raster_path = str(
            working_dir / f"dt_{condition_id}_{base_raster_path.name}"
        )
        create_distance_transform(condition_raster_path, distance_transform_raster_path)
        maxdist_buffered_ds_coverage_raster_path = str(
            working_dir
            / Path(
                f"max_dist_{max_downstream_distance_m}_"
                + (Path(buffered_ds_coverage_raster_path)).name
            )
        )
        geoprocessing.raster_calculator(
            [
                (buffered_ds_coverage_raster_path, 1),
                (distance_transform_raster_path, 1),
            ],
            _distance_mask_op,
            maxdist_buffered_ds_coverage_raster_path,
            gdal.GDT_Byte,
            None,
            calc_raster_stats=False,
            raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
        )
        buffered_ds_coverage_raster_path = maxdist_buffered_ds_coverage_raster_path

    def coverage_mask_op(mask):
        # mask values come from convolve so they can be veeeeeery close
        # to 0 without being 0 when the should be, so we just cap that here
        return (mask > DOWNSTREAM_COVERAGE_EPSILON).astype(np.uint8)

    geoprocessing.raster_calculator(
        [(str(buffered_ds_coverage_raster_path), 1)],
        coverage_mask_op,
        target_coverage_raster_path,
        gdal.GDT_Byte,
        0,
        calc_raster_stats=False,
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )
    logger.info(
        "%s finished in %.1fs; rss_delta=%s; %s",
        worker_label,
        time.monotonic() - worker_start_time,
        _format_bytes(_process_tree_rss_bytes() - worker_start_rss),
        _memory_status_text(),
    )
    return target_coverage_raster_path


def calc_flow_dir(dem_path, working_dir, target_flow_dir_raster_path):
    configure_gdal_cache()
    logger = logging.getLogger(__name__)
    worker_label = f"flow direction worker {Path(target_flow_dir_raster_path).stem}"
    worker_start_time = time.monotonic()
    worker_start_rss = _process_tree_rss_bytes()
    logger.info("%s started; %s", worker_label, _memory_status_text())
    pit_filled_raster_path = working_dir / f"pit_filled_{Path(dem_path).name}"
    routing.fill_pits(
        (dem_path, 1),
        pit_filled_raster_path,
        working_dir=working_dir,
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )
    routing.flow_dir_mfd(
        (str(pit_filled_raster_path), 1),
        str(target_flow_dir_raster_path),
        working_dir=str(working_dir),
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE,
    )
    logger.info(
        "%s finished in %.1fs; rss_delta=%s; %s",
        worker_label,
        time.monotonic() - worker_start_time,
        _format_bytes(_process_tree_rss_bytes() - worker_start_rss),
        _memory_status_text(),
    )


def _raster_bounds_in_crs(raster: rasterio.DatasetReader, target_crs) -> tuple:
    """Return raster bounds transformed into a target CRS.

    Args:
        raster: Open Rasterio dataset whose bounds should be transformed.
        target_crs: Coordinate reference system accepted by Rasterio for the
            returned bounds.

    Returns:
        Tuple of ``(left, bottom, right, top)`` bounds in ``target_crs``.

    Raises:
        ValueError: If ``raster`` has no CRS.
    """
    if raster.crs is None:
        raise ValueError(f"Raster has no CRS and cannot be combined: {raster.name}")
    return transform_bounds(
        raster.crs,
        target_crs,
        *raster.bounds,
        densify_pts=21,
    )


def _combined_raster_profile(
    raster_path_list: list[str],
    wgs84_pixel_size: float,
    dtype: str = "float32",
    nodata=None,
) -> dict:
    """Build a tiled WGS84 Rasterio profile covering all input rasters.

    Args:
        raster_path_list: Paths to rasters whose combined bounds define the
            output extent.
        wgs84_pixel_size: Output pixel size in WGS84 degrees. The absolute
            value is used so callers may pass a signed pixel size.
        dtype: Rasterio dtype for the output profile.
        nodata: Nodata value for the output profile.

    Returns:
        Rasterio profile for a single-band, 256 x 256 tiled GeoTIFF in
        EPSG:4326.

    Raises:
        ValueError: If ``wgs84_pixel_size`` is not positive, no finite raster
            bounds are found, or the calculated output dimensions are invalid.
    """
    pixel_size = abs(float(wgs84_pixel_size))
    if pixel_size <= 0:
        raise ValueError(f"wgs84_pixel_size must be positive, got {wgs84_pixel_size}")

    target_crs = "EPSG:4326"
    minx = miny = math.inf
    maxx = maxy = -math.inf

    for raster_path in raster_path_list:
        with rasterio.open(raster_path) as raster:
            left, bottom, right, top = _raster_bounds_in_crs(raster, target_crs)
        minx = min(minx, left)
        miny = min(miny, bottom)
        maxx = max(maxx, right)
        maxy = max(maxy, top)

    if not all(math.isfinite(value) for value in [minx, miny, maxx, maxy]):
        raise ValueError("No valid raster bounds were found for population combine.")

    minx = math.floor(minx / pixel_size) * pixel_size
    miny = math.floor(miny / pixel_size) * pixel_size
    maxx = math.ceil(maxx / pixel_size) * pixel_size
    maxy = math.ceil(maxy / pixel_size) * pixel_size

    width = int(math.ceil((maxx - minx) / pixel_size))
    height = int(math.ceil((maxy - miny) / pixel_size))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid combined raster dimensions from bounds "
            f"{[minx, miny, maxx, maxy]}: {width}x{height}"
        )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": target_crs,
        "transform": from_origin(minx, maxy, pixel_size, pixel_size),
        "nodata": nodata,
    }
    _set_tiled_geotiff_creation_options(profile)
    return profile


def _integer_window(raw_window: Window, width: int, height: int) -> Window:
    """Round a Rasterio window outward and clamp it to raster dimensions.

    Args:
        raw_window: Window with possibly fractional offsets or dimensions.
        width: Raster width in pixels used to clamp the right edge.
        height: Raster height in pixels used to clamp the bottom edge.

    Returns:
        Window with integer offsets and dimensions that stays within the raster
        bounds.
    """
    col_start = max(0, math.floor(raw_window.col_off))
    row_start = max(0, math.floor(raw_window.row_off))
    col_stop = min(width, math.ceil(raw_window.col_off + raw_window.width))
    row_stop = min(height, math.ceil(raw_window.row_off + raw_window.height))
    return Window(
        col_start,
        row_start,
        max(0, col_stop - col_start),
        max(0, row_stop - row_start),
    )


def _iter_block_windows(window: Window, block_size: int = RASTER_BLOCK_SIZE):
    """Yield block-sized windows within a larger window.

    Args:
        window: Parent window to split into smaller read/write chunks.
        block_size: Maximum width and height of each yielded window in pixels.

    Yields:
        Window objects covering ``window`` without exceeding ``block_size`` in
        either dimension.
    """
    row_start = int(window.row_off)
    col_start = int(window.col_off)
    row_stop = int(window.row_off + window.height)
    col_stop = int(window.col_off + window.width)
    for row_off in range(row_start, row_stop, block_size):
        block_height = min(block_size, row_stop - row_off)
        for col_off in range(col_start, col_stop, block_size):
            block_width = min(block_size, col_stop - col_off)
            yield Window(col_off, row_off, block_width, block_height)


def _rasterize_wgs84_bounds_mask(
    bounds_wgs84,
    target_transform,
    target_crs,
    height: int,
    width: int,
) -> np.ndarray:
    """Create a binary mask from WGS84 bounds using target pixel centers.

    This avoids projecting and rasterizing a near-global WGS84 polygon into a
    local projected CRS, which can collapse the rasterized geometry near the
    projection seam.
    """
    minx, miny, maxx, maxy = bounds_wgs84
    transformer = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    mask_array = np.zeros((height, width), dtype=np.int8)
    full_window = Window(0, 0, width, height)
    for window in _iter_block_windows(full_window):
        row_start = int(window.row_off)
        row_stop = row_start + int(window.height)
        col_start = int(window.col_off)
        col_stop = col_start + int(window.width)
        rows = np.arange(row_start, row_stop, dtype=np.float64) + 0.5
        cols = np.arange(col_start, col_stop, dtype=np.float64) + 0.5
        col_grid, row_grid = np.meshgrid(cols, rows)
        x_grid, y_grid = target_transform * (col_grid, row_grid)
        lon_grid, lat_grid = transformer.transform(x_grid, y_grid)
        valid_lon_lat = np.isfinite(lon_grid) & np.isfinite(lat_grid)
        if minx <= maxx:
            inside_lon = (lon_grid >= minx) & (lon_grid <= maxx)
        else:
            inside_lon = (lon_grid >= minx) | (lon_grid <= maxx)
        inside_lat = (lat_grid >= miny) & (lat_grid <= maxy)
        mask_array[row_start:row_stop, col_start:col_stop] = (
            valid_lon_lat & inside_lon & inside_lat
        ).astype(np.int8)
    return mask_array


def _create_zeroed_raster(
    target_path: Path,
    profile: dict,
    desc: str = "initialize combined raster",
) -> None:
    """Create a zero-filled raster for later windowed max writes.

    Args:
        target_path: Path where the output raster should be created.
        profile: Rasterio profile describing the target raster. The profile is
            expected to include the 256 x 256 tiled GeoTIFF creation options.
        desc: Progress-bar description for the initialization pass.

    Returns:
        None.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(profile["dtype"])
    block_count = math.ceil(profile["width"] / RASTER_BLOCK_SIZE) * math.ceil(
        profile["height"] / RASTER_BLOCK_SIZE
    )
    with rasterio.open(target_path, "w", **profile) as target:
        for _, window in tqdm(
            target.block_windows(1),
            total=block_count,
            desc=desc,
            unit="block",
        ):
            target.write(
                np.zeros((int(window.height), int(window.width)), dtype=dtype),
                1,
                window=window,
            )


def stitch_coverage_masks(
    coverage_id_raster_list,
    wgs84_pixel_size,
    working_dir,
    target_coverage_raster_path,
):
    """Stitch binary coverage rasters into one global binary coverage raster."""
    logger = logging.getLogger(__name__)

    raster_count = len(coverage_id_raster_list)
    if raster_count == 0:
        raise ValueError("No coverage rasters were provided for stitching.")

    logger.info(
        "stitching %d coverage rasters into %s",
        raster_count,
        target_coverage_raster_path,
    )

    coverage_ids = [coverage_id for coverage_id, _ in coverage_id_raster_list]
    coverage_raster_list = [str(path) for _, path in coverage_id_raster_list]
    target_coverage_raster_path = Path(target_coverage_raster_path)
    target_profile = _combined_raster_profile(
        coverage_raster_list,
        wgs84_pixel_size,
        dtype="uint8",
        nodata=0,
    )

    _create_zeroed_raster(target_coverage_raster_path, target_profile)

    stitch_state = {
        "index": 0,
        "raster_id": "",
        "pixels": 0,
    }
    coverage_counts_by_id = {}

    with _log_heartbeat(
        logger,
        lambda: (
            "stitching coverage rasters: "
            f"{stitch_state['index']}/{raster_count} rasters, "
            f"current={stitch_state['raster_id']}, "
            f"{stitch_state['pixels']} target pixels visited"
        ),
    ):
        with rasterio.open(target_coverage_raster_path, "r+") as target:
            target_crs = target.crs
            target_transform = target.transform
            with tqdm(
                zip(coverage_ids, coverage_raster_list),
                total=raster_count,
                desc="stitch coverage rasters",
                unit="raster",
            ) as progress:
                for raster_index, (coverage_id, source_path) in enumerate(
                    progress,
                    start=1,
                ):
                    stitch_state["index"] = raster_index - 1
                    stitch_state["raster_id"] = str(coverage_id)
                    source_count = np.float64(0)
                    with rasterio.open(source_path) as source:
                        source_bounds = _raster_bounds_in_crs(source, target_crs)
                        target_window = _integer_window(
                            from_bounds(*source_bounds, transform=target_transform),
                            target.width,
                            target.height,
                        )
                        if target_window.width == 0 or target_window.height == 0:
                            coverage_counts_by_id[coverage_id] = source_count
                            continue

                        vrt_kwargs = {
                            "crs": target_crs,
                            "transform": target_transform,
                            "width": target.width,
                            "height": target.height,
                            "resampling": Resampling.nearest,
                            "nodata": 0,
                        }
                        if source.nodata is not None:
                            vrt_kwargs["src_nodata"] = source.nodata

                        with WarpedVRT(source, **vrt_kwargs) as source_vrt:
                            for window in _iter_block_windows(target_window):
                                incoming = source_vrt.read(1, window=window)
                                incoming = (incoming > 0).astype(np.uint8)
                                stitch_state["pixels"] += int(
                                    window.width * window.height
                                )
                                source_count += np.sum(incoming, dtype=np.float64)
                                if not np.any(incoming):
                                    continue
                                existing = target.read(1, window=window)
                                np.maximum(existing, incoming, out=existing)
                                target.write(existing, 1, window=window)

                    coverage_counts_by_id[coverage_id] = source_count
                    stitch_state["index"] = raster_index
                    progress.set_postfix_str(str(coverage_id))

    coverage_counts_by_id["coverage"] = _sum_raster_blocks(target_coverage_raster_path)
    logger.info("wrote coverage raster %s", target_coverage_raster_path)
    return coverage_counts_by_id


def mask_population_with_coverage(
    population_raster_path,
    coverage_raster_path,
    target_population_raster_path,
):
    """Mask global population values by a stitched coverage raster.

    The coverage raster defines the target grid for the output. Population is
    read through a WarpedVRT on that grid, with nodata and negative values
    treated as 0. Pixels where coverage is 0 are written as 0, and pixels where
    coverage is positive retain their population value.

    Args:
        population_raster_path: Path-like global population raster to mask.
        coverage_raster_path: Path-like binary coverage raster where values
            greater than 0 indicate covered pixels.
        target_population_raster_path: Path-like output raster where masked
            population values will be written.

    Returns:
        Sum of the output masked population raster.
    """
    logger = logging.getLogger(__name__)
    target_population_raster_path = Path(target_population_raster_path)
    target_population_raster_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(coverage_raster_path) as coverage:
        with rasterio.open(population_raster_path) as population:
            target_profile = coverage.profile.copy()
            target_profile.update(
                {
                    "dtype": population.dtypes[0],
                    "count": 1,
                    "nodata": 0,
                }
            )
            _set_tiled_geotiff_creation_options(target_profile)

            vrt_kwargs = {
                "crs": coverage.crs,
                "transform": coverage.transform,
                "width": coverage.width,
                "height": coverage.height,
                "resampling": Resampling.nearest,
                "nodata": POPULATION_COMBINE_VRT_NODATA,
            }
            if population.nodata is not None:
                vrt_kwargs["src_nodata"] = population.nodata

            population_sum = np.float64(0)
            with WarpedVRT(population, **vrt_kwargs) as population_vrt:
                with rasterio.open(
                    target_population_raster_path, "w", **target_profile
                ) as target:
                    for _, window in tqdm(
                        coverage.block_windows(1),
                        desc=f"mask population {Path(coverage_raster_path).stem}",
                        unit="block",
                    ):
                        coverage_block = coverage.read(1, window=window)
                        population_block = population_vrt.read(1, window=window)
                        population_block = population_block.astype(
                            target_profile["dtype"],
                            copy=False,
                        )
                        population_block[population_block < 0] = 0
                        masked_population = np.where(
                            (coverage_block > 0) & (population_block > 0),
                            population_block,
                            0,
                        )
                        population_sum += np.sum(
                            masked_population,
                            dtype=np.float64,
                        )
                        target.write(masked_population, 1, window=window)

    logger.info(
        "wrote population raster %s: %.6g",
        target_population_raster_path,
        population_sum,
    )
    return float(population_sum)


def calculate_taskgraph_worker_count(config: dict, work_unit_count: int) -> int:
    """Calculate a TaskGraph worker count from available workflow fanout.

    Args:
        config: Normalized workflow configuration returned by
            ``process_config``.
        work_unit_count: Number of independent AOI or drain-partition work
            units that will be processed.

    Returns:
        Worker count bounded by the physical CPU count. Conditional routing
        workflows default to a conservative worker count because DEM routing
        and raster warping are memory-heavy.
    """
    physical_cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    configured_workers = config.get("inputs", {}).get("taskgraph_workers")
    if configured_workers is not None:
        return min(int(configured_workers), physical_cpu_count)

    has_travel_time_mask = any(
        mask.get("type") == "travel_time_population" for mask in config.get("masks", [])
    )
    has_conditional_mask = any(
        mask.get("type") == "conditional_raster" for mask in config.get("masks", [])
    )

    if has_conditional_mask:
        desired_worker_count = min(
            DEFAULT_CONDITIONAL_TASKGRAPH_WORKERS,
            max(1, work_unit_count),
        )
    elif has_travel_time_mask:
        desired_worker_count = min(
            DEFAULT_TRAVEL_TIME_TASKGRAPH_WORKERS,
            max(1, work_unit_count),
        )
    else:
        desired_worker_count = max(1, work_unit_count)
    if has_travel_time_mask:
        desired_worker_count = max(2, desired_worker_count)
    return min(desired_worker_count, physical_cpu_count)


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description="Extract and normalize analysis config from YAML."
    )
    ap.add_argument("config", type=Path, help="Path to YAML config file")
    ap.add_argument(
        "--validate-paths",
        action="store_true",
        help="Lightly validate paths exist (non-glob)",
    )
    args = ap.parse_args()

    config = process_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config["logging"]["level"], config["logging"]["to_file"])
    gdal_cachemax_mb = configure_gdal_cache(config["inputs"]["gdal_cachemax_mb"])
    logger.info("using GDAL cache max of %d MiB per process", gdal_cachemax_mb)

    validate_paths(config)
    logger.info(f"{args.config} read successfully")

    aoi_id_to_path = collect_aoi_files(config)
    logger.info(f"found {len(aoi_id_to_path)} aois to process")

    wgs84_pixel_size = config["inputs"]["wgs84_pixel_size"]
    has_conditional_mask = any(
        mask.get("type") == "conditional_raster" for mask in config.get("masks", [])
    )
    debug_drain_index = config["inputs"].get("debug_drain_index")

    aoi_work_items = {}
    for aoi_key, aoi_vector_path in aoi_id_to_path.items():
        working_dir = Path(config["work_dir"]) / Path(aoi_key)
        working_dir.mkdir(parents=True, exist_ok=True)
        picked_crs = choose_equidistant_crs_from_bbox(aoi_vector_path)
        if has_conditional_mask:
            partition_paths = partition_subwatersheds_by_terminal_drain(
                aoi_vector_path,
                config["inputs"]["subwatershed_vector_path"],
                picked_crs,
                working_dir / "drain_partitions",
                debug_drain_index=debug_drain_index,
            )
            if debug_drain_index is not None:
                selected_partition_id = next(iter(partition_paths))
                logger.info(
                    "keeping debug intermediates for %s %s in %s",
                    aoi_key,
                    selected_partition_id,
                    working_dir / selected_partition_id,
                )
                logger.debug(
                    "using %s as the only drain partition for debug_drain_index=%d",
                    partition_paths[selected_partition_id],
                    debug_drain_index,
                )
        else:
            partition_paths = {}
        aoi_work_items[aoi_key] = {
            "aoi_vector_path": aoi_vector_path,
            "target_crs": picked_crs,
            "working_dir": working_dir,
            "partition_paths": partition_paths,
        }
        if has_conditional_mask:
            logger.info(
                "found %d drain partitions for %s",
                len(partition_paths),
                aoi_key,
            )

    work_unit_count = sum(
        len(item["partition_paths"]) for item in aoi_work_items.values()
    )
    if not has_conditional_mask:
        work_unit_count = len(aoi_work_items)
    n_workers = calculate_taskgraph_worker_count(config, work_unit_count)
    update_rate = None  # Supress taskgraph output
    logger.info("using %d TaskGraph workers", n_workers)

    task_graph = taskgraph.TaskGraph(config["work_dir"], n_workers, update_rate)
    population_result_ids = set()
    union_population_id = "union_population"
    population_result_ids.add(union_population_id)
    population_results = collections.defaultdict(dict)
    task_progress_paths = []
    for aoi_key, aoi_info in aoi_work_items.items():
        aoi_vector_path = aoi_info["aoi_vector_path"]
        working_dir = aoi_info["working_dir"]
        dem_raster_path = config["inputs"]["dem_raster_path"]

        partition_contexts = {}
        if has_conditional_mask:
            for partition_id, partition_vector_path in aoi_info[
                "partition_paths"
            ].items():
                partition_working_dir = working_dir / partition_id
                partition_working_dir.mkdir(parents=True, exist_ok=True)
                clipped_dem_path = str(
                    partition_working_dir / f"{Path(dem_raster_path).stem}_clipped.tif"
                )
                dem_clip_task = task_graph.add_task(
                    func=align_and_resize_raster_on_vector,
                    args=(
                        dem_raster_path,
                        clipped_dem_path,
                        "near",
                        [wgs84_pixel_size, -wgs84_pixel_size],
                        partition_vector_path,
                    ),
                    target_path_list=[clipped_dem_path],
                    task_name=f"clip DEM for {aoi_key} {partition_id}",
                )
                dem_path_root, dem_path_ext = os.path.splitext(str(clipped_dem_path))
                target_flow_dir_raster_path = f"{dem_path_root}_mfdflow{dem_path_ext}"
                flow_dir_task = task_graph.add_task(
                    func=calc_flow_dir,
                    args=(
                        clipped_dem_path,
                        partition_working_dir,
                        target_flow_dir_raster_path,
                    ),
                    dependent_task_list=[dem_clip_task],
                    target_path_list=[target_flow_dir_raster_path],
                    task_name=f"calculate flow dir for {aoi_key} {partition_id}",
                )
                partition_contexts[partition_id] = {
                    "vector_path": partition_vector_path,
                    "flow_dir_raster_path": Path(target_flow_dir_raster_path),
                    "flow_dir_task": flow_dir_task,
                    "working_dir": partition_working_dir,
                }
            if not partition_contexts:
                raise ValueError(
                    f"No drain partitions were found for conditional AOI {aoi_key}."
                )
        else:
            partition_contexts[aoi_key] = {
                "vector_path": aoi_vector_path,
                "flow_dir_raster_path": None,
                "flow_dir_task": None,
                "working_dir": working_dir,
            }

        coverage_raster_tasks = {}
        coverage_id_raster_list = []
        for mask_section in config["masks"]:
            section_id = mask_section["id"]
            population_result_ids.add(f"{section_id}_population")
            target_coverage_raster_path = (
                output_dir / f"{aoi_key}_{section_id}_coverage.tif"
            )
            partition_coverage_id_raster_list = []
            partition_coverage_tasks = []
            if mask_section["type"] == "travel_time_population":
                for partition_id, partition_context in partition_contexts.items():
                    partition_coverage_raster_path = (
                        partition_context["working_dir"] / f"{section_id}_coverage.tif"
                    )
                    task_progress_paths.append(partition_coverage_raster_path)
                    partition_coverage_id_raster_list.append(
                        (partition_id, partition_coverage_raster_path)
                    )
                    travel_time_working_dir = (
                        partition_context["working_dir"] / f"{section_id}_workdir"
                    )
                    use_wgs84_bounds_mask = (
                        aoi_key == FULL_RASTER_EXTENT_AOI_ID
                        and not has_conditional_mask
                        and debug_drain_index is None
                    )
                    travel_task = task_graph.add_task(
                        func=calculate_travel_time_coverage,
                        args=(
                            config["inputs"]["traveltime_raster_path"],
                            partition_context["vector_path"],
                            mask_section["params"]["max_hours"],
                            config["inputs"]["travel_time_pixel_size_m"],
                            partition_coverage_raster_path,
                            travel_time_working_dir,
                            use_wgs84_bounds_mask,
                        ),
                        target_path_list=[partition_coverage_raster_path],
                        task_name=(
                            f"travel-time coverage {section_id} "
                            f"for {aoi_key} {partition_id}"
                        ),
                    )
                    partition_coverage_tasks.append(travel_task)
            elif mask_section["type"] == "conditional_raster":
                for (
                    partition_id,
                    partition_context,
                ) in partition_contexts.items():
                    if partition_context["flow_dir_task"] is None:
                        raise ValueError(
                            "conditional_raster masks require drain partitions "
                            "and flow-direction rasters."
                        )
                    partition_coverage_raster_path = (
                        partition_context["working_dir"] / f"{section_id}_coverage.tif"
                    )
                    task_progress_paths.append(partition_coverage_raster_path)
                    partition_coverage_id_raster_list.append(
                        (partition_id, partition_coverage_raster_path)
                    )
                    conditional_task = task_graph.add_task(
                        func=calculate_downstream_coverage_from_conditional_raster,
                        args=(
                            partition_context["vector_path"],
                            partition_context["flow_dir_raster_path"],
                            Path(mask_section["params"]["condition_raster_path"]),
                            section_id,
                            mask_section["params"]["expression"],
                            config["inputs"]["buffer_size_m"],
                            mask_section["params"].get(
                                "max_downstream_distance_m", None
                            ),
                            config["inputs"]["travel_time_pixel_size_m"],
                            partition_context["working_dir"],
                            partition_coverage_raster_path,
                        ),
                        dependent_task_list=[partition_context["flow_dir_task"]],
                        target_path_list=[partition_coverage_raster_path],
                        task_name=(
                            f"conditional downstream {section_id} "
                            f"for {aoi_key} {partition_id}"
                        ),
                    )
                    partition_coverage_tasks.append(conditional_task)
            else:
                raise ValueError(f"unknown mask section type: {mask_section['type']}")

            section_coverage_task = task_graph.add_task(
                func=stitch_coverage_masks,
                args=(
                    partition_coverage_id_raster_list,
                    wgs84_pixel_size,
                    working_dir / f"stitch_{section_id}_coverage",
                    target_coverage_raster_path,
                ),
                dependent_task_list=partition_coverage_tasks,
                target_path_list=[target_coverage_raster_path],
                task_name=f"stitch coverage for {aoi_key} {section_id}",
            )
            coverage_raster_tasks[section_id] = section_coverage_task
            task_progress_paths.append(target_coverage_raster_path)
            coverage_id_raster_list.append((section_id, target_coverage_raster_path))

            target_population_raster_path = (
                output_dir / f"{aoi_key}_{section_id}_population.tif"
            )
            population_task = task_graph.add_task(
                func=mask_population_with_coverage,
                args=(
                    config["inputs"]["population_raster_path"],
                    target_coverage_raster_path,
                    target_population_raster_path,
                ),
                dependent_task_list=[section_coverage_task],
                store_result=True,
                target_path_list=[target_population_raster_path],
                task_name=f"mask population for {aoi_key} {section_id}",
            )
            population_results[aoi_key][f"{section_id}_population"] = population_task
            task_progress_paths.append(target_population_raster_path)

        target_union_coverage_raster_path = output_dir / f"{aoi_key}_union_coverage.tif"
        task_progress_paths.append(target_union_coverage_raster_path)
        union_coverage_task = task_graph.add_task(
            func=stitch_coverage_masks,
            args=(
                coverage_id_raster_list,
                wgs84_pixel_size,
                working_dir / "stitch_union_coverage",
                target_union_coverage_raster_path,
            ),
            dependent_task_list=list(coverage_raster_tasks.values()),
            target_path_list=[target_union_coverage_raster_path],
            task_name=f"stitch union coverage for {aoi_key}",
        )

        target_union_population_raster_path = (
            output_dir / f"{aoi_key}_union_population.tif"
        )
        union_population_task = task_graph.add_task(
            func=mask_population_with_coverage,
            args=(
                config["inputs"]["population_raster_path"],
                target_union_coverage_raster_path,
                target_union_population_raster_path,
            ),
            dependent_task_list=[union_coverage_task],
            store_result=True,
            target_path_list=[target_union_population_raster_path],
            task_name=f"mask population for {aoi_key} union",
        )
        population_results[aoi_key][union_population_id] = union_population_task
        task_progress_paths.append(target_union_population_raster_path)

    task_graph.close()
    def _task_progress_message() -> str:
        complete_count = 0
        for target_path in task_progress_paths:
            try:
                if Path(target_path).exists() and Path(target_path).stat().st_size > 0:
                    complete_count += 1
            except OSError:
                pass
        return (
            "workflow tasks running: "
            f"{complete_count}/{len(task_progress_paths)} target rasters present; "
            f"{_memory_status_text()}"
        )

    with _log_heartbeat(logger, _task_progress_message):
        task_graph.join()
    rows = []
    for aoi_key, results in population_results.items():
        row = {"aoi": aoi_key}
        for header in population_result_ids:
            row[header] = results[header].get() if header in results else "n/a"
        rows.append(row)

    df = pd.DataFrame(rows, columns=["aoi"] + list(population_result_ids))
    cols = (
        ["aoi"]
        + [c for c in df.columns if c not in ("aoi", union_population_id)]
        + [union_population_id]
    )
    df = df[cols]
    csv_path = (
        output_dir
        / f'{config["run_name"]}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'
    )
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
