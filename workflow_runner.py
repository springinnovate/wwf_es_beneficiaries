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
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import os
import glob
import logging
import sys
from itertools import islice

from ecoshard import taskgraph
import psutil
import yaml

from rasterio.warp import calculate_default_transform, reproject, Resampling
from ecoshard import geoprocessing
from ecoshard.geoprocessing import routing
from osgeo import gdal, osr
from shapely.geometry import box
from shapely.ops import transform
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
import pyogrio
from pyproj import CRS, Transformer, Geod

import shortest_distances

logging.getLogger("rasterio").setLevel(logging.WARNING)


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
        options=(
            "TILED=YES",
            "BIGTIFF=YES",
            "COMPRESS=LZW",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "NUM_THREADS=ALL_CPUS",
        ),
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
            f"Azimuthal Equidistant fallback - "
            f"large N–S extent {ns_km:.0f} km"
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
                'subwatershed_vector_path', or 'aoi_vector_pattern'.
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
    aoi_vector_pattern = _as_list(inputs.get("aoi_vector_pattern", []))

    wgs84_pixel_size = inputs.get("wgs84_pixel_size", None)
    travel_time_pixel_size_m = inputs.get("travel_time_pixel_size_m", None)
    buffer_size_m = inputs.get("buffer_size_m", None)

    missing_messages = []
    if not population_raster_path:
        missing_messages.append(
            "population_raster_path (path to population raster)"
        )
    elif not Path(population_raster_path).exists():
        missing_messages.append(
            f"population_raster_path does not exist: {population_raster_path}"
        )

    if not traveltime_raster_path:
        missing_messages.append(
            "traveltime_raster_path (path to travel-time raster)"
        )
    elif not Path(traveltime_raster_path).exists():
        missing_messages.append(
            f"traveltime_raster_path does not exist: {traveltime_raster_path}"
        )
        missing_messages.append(
            "population_raster_path (path to population raster)"
        )

    if not dem_raster_path:
        missing_messages.append("dem_raster_path (path to DEM raster)")
    elif not Path(dem_raster_path).exists():
        missing_messages.append(
            f"dem_raster_path does not exist: {dem_raster_path}"
        )

    if not subwatershed_vector_path:
        missing_messages.append(
            "subwatershed_vector_path (path to subwatershed shapefile/vector)"
        )
    elif not Path(subwatershed_vector_path).exists():
        missing_messages.append(
            f"subwatershed_vector_path does not exist: {subwatershed_vector_path}"
        )

    if not aoi_vector_pattern:
        missing_messages.append(
            "aoi_vector_pattern (one or more AOI file patterns)"
        )

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
        msg = "Missing required input(s):\n  - " + "\n  - ".join(
            missing_messages
        )
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
                f"sections[{idx}] must contain at least one of "
                f'["masks", "combine"]'
            )
    if len(found_sections) != 2:
        raise ValueError(
            "Expected both a `masks` and `combine` section but missing at "
            "least one."
        )

    if errors:
        raise ValueError("Invalid sections:\n  - " + "\n  - ".join(errors))
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
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    fmt = "%(asctime)s %(filename)s:%(lineno)d [%(levelname)s]  %(message)s"

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    return root


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
        if isinstance(path_like, str) and any(
            ch in path_like for ch in ["*", "?", "["]
        ):
            # skip globs
            return
        if isinstance(path_like, str) and not Path(path_like).exists():
            issues.append((label, f"not found: {path_like}"))

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

    if issues:
        formatted = "\n".join(f"- {label}: {msg}" for label, msg in issues)
        raise ValueError(
            f"Path validation failed with {len(issues)} issue(s):\n{formatted}"
        )


def print_yaml_config(config):
    """Just for debugging..."""
    logger = logging.getLogger(__name__)
    logger.info("doing something important")
    logger.info("run_name: %s", config["run_name"])
    logger.info("work_dir: %s", config["work_dir"])
    logger.info("output_dir: %s", config["output_dir"])
    logger.info("inputs:")
    for k, v in config["inputs"].items():
        if isinstance(v, list):
            logger.info("  %s:", k)
            for item in v:
                logger.info("    - %s", item)
        else:
            logger.info("  %s: %s", k, v)
    logger.info("masks:")
    for m in config["masks"]:
        logger.info("  - id: %s", m["id"])
        logger.info("    type: %s", m["type"])
        if m.get("params"):
            logger.info("    params:")
            for pk, pv in m["params"].items():
                logger.info("      %s: %s", pk, pv)
    logger.info("combine:")
    for c in config["combine"]:
        logger.info("  - %s", c)
    logger.info("logging:")
    logger.info("  level: %s", config["logging"]["level"])
    logger.info("  to_file: %s", config["logging"]["to_file"])


def collect_aoi_files(config: dict) -> dict[str, Path]:
    """Collect AOI files from the patterns.

    In config['inputs']['aoi_vector_pattern'].

    Returns:
        dict mapping file stem -> Path

    Raises:
        ValueError if two or more files share the same stem.
    """
    inputs = config.get("inputs", {})
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
    return aoi_map


def _chunks(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])


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
    logger.info(f"processing aoi {aoi_vector_path}")

    # Read AOI and precompute union + bbox; repair invalid with make_valid if available
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

    attrs = pyogrio.read_dataframe(
        subwatershed_vector_path,
        columns=["HYBAS_ID", "NEXT_DOWN"],
        read_geometry=False,
    )
    hybas_to_nextdown = dict(
        zip(attrs["HYBAS_ID"].to_numpy(), attrs["NEXT_DOWN"].to_numpy())
    )

    logger.info(f"spatial pre-filter {aoi_vector_path} w/ bb")
    sub_bbox_gdf = pyogrio.read_dataframe(
        subwatershed_vector_path,
        bbox=aoi_bbox.bounds,
        columns=["HYBAS_ID", "NEXT_DOWN", "geometry"],
    )
    if sub_bbox_gdf.empty:
        raise ValueError(f"No candidates found in bbox for {aoi_vector_path}.")

    hits = sub_bbox_gdf.sindex.query(aoi_union, predicate="intersects")
    if len(hits) == 0:
        raise ValueError(
            f"No intersecting sub-watersheds for {aoi_vector_path}."
        )

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
    logger.info(f"fetch geometries by attribute filter {aoi_vector_path}")
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
            sub_gdf.loc[invalid, "geometry"] = sub_gdf.loc[
                invalid, "geometry"
            ].buffer(0)

    # Reproject to AOI CRS and write
    if sub_crs and aoi_crs and sub_crs != aoi_crs:
        sub_gdf = sub_gdf.to_crs(aoi_crs)

    sub_gdf = sub_gdf.to_crs(target_crs.crs)
    sub_gdf.to_file(target_subset_subwatersheds_vector_path, driver="GPKG")
    logger.info(f"all done subwatershedding {aoi_vector_path}")


def align_and_resize_raster_stack_on_vector(
    raster_path_list,
    target_path_list,
    resample_method_list,
    target_pixel_size,
    bounding_vector_path,
):
    gdf = gpd.read_file(bounding_vector_path)
    gdf = gdf.set_geometry(gdf.geometry.make_valid())
    if not gdf.crs or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # minx, miny, maxx, maxy = gdf.total_bounds
    geoprocessing.align_and_resize_raster_stack(
        raster_path_list,
        target_path_list,
        resample_method_list,
        target_pixel_size,
        [float(x) for x in gdf.total_bounds],
    )


def eck4_limits(r=6371000):
    crs_eck4 = CRS.from_proj4(f"+proj=eck4 +R={r} +units=m +no_defs")
    T_fwd = Transformer.from_crs("EPSG:4326", crs_eck4, always_xy=True)
    xs, ys = T_fwd.transform([-180, 180, 0, 0], [0, 0, 90, -90])
    return abs(xs[0]), abs(ys[2])  # xmax, ymax


ECKERT_IV_MAX_X, ECKERT_IV_MAX_Y = eck4_limits()  # ≈ 15 110 000 , 7 540 000


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
    if "+proj=eck4" not in bbox_gdf.crs.to_proj4().lower():
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

        if "+proj=eck4" in bbox_gdf.crs.to_proj4().lower():
            logger.info(
                "eckert is so broken, just doing regular lat/lng bounds"
            )
            projected_box_gdf = transform_edge_points_eckert_to_wgs84(bbox_gdf)
            projected_box_gdf = gpd.GeoDataFrame(
                geometry=[box(-179, -80, 179, 80)], crs="EPSG:4326"
            )
        else:
            # Check intermediate clamped bounds explicitly:
            logger.info(f"Clamped bbox bounds: {bbox_gdf.total_bounds}")

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

        with rasterio.open(target_raster_path, "w", **out_meta) as dst:
            reproject(
                source=out_image,
                destination=rasterio.band(dst, 1),
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )


def apply_travel_time_mask(
    traveltime_raster_path: Path,
    population_raster_path: Path,
    aoi_vector_path: Path,
    max_hours: float,
    target_crs: str,
    target_pop_raster_path: Path,
    working_dir: Path,
):
    """
    Create a travel-time mask clipped to an AOI.

    Args:
        traveltime_raster_path (str | Path): Path to a raster where each pixel
            encodes travel time (hours) to a target destination.
        aoi_vector_path (str | Path): Path to a vector dataset (e.g. Shapefile,
            GeoPackage) defining the area of interest to clip to.
        max_hours (float): Maximum travel time in hours to include.

    Returns:
        rasterio.io.DatasetReader: An in-memory raster mask where pixels within
        the AOI and <= max_hours are valid (1), and others are nodata (0).
    """
    logger = logging.getLogger(__name__)
    max_time_mins = max_hours * 60

    aoi_vector = gpd.read_file(aoi_vector_path)
    projected_gdf = aoi_vector.to_crs(target_crs)
    bbox = projected_gdf.total_bounds
    buffer_distance_m = (
        max_hours * 104 * 1000
    )  # drive 65mph for that many hours

    buffered_bbox = box(
        bbox[0] - buffer_distance_m,
        bbox[1] - buffer_distance_m,
        bbox[2] + buffer_distance_m,
        bbox[3] + buffer_distance_m,
    )

    logger.info(f"buffered box: {buffered_bbox}")
    bbox_gdf = gpd.GeoDataFrame(
        {"geometry": [buffered_bbox]}, crs=projected_gdf.crs
    )

    target_pop_clipped_raster_path = Path(
        working_dir / f"{traveltime_raster_path.stem}_travel_clip.tif"
    )
    target_friction_clipped_raster_path = Path(
        working_dir / f"{traveltime_raster_path.stem}_travel_clip.tif"
    )
    target_aoi_raster_path = Path(working_dir / "travel_time_aoi_mask.tif")

    _clip_and_reproject_raster(
        population_raster_path,
        bbox_gdf,
        projected_gdf.crs,
        target_pop_clipped_raster_path,
    )

    with rasterio.open(target_pop_clipped_raster_path) as pop_ref:
        ref_meta = pop_ref.meta.copy()
        pop_array = pop_ref.read(1).astype(np.int64)

    _clip_and_reproject_raster(
        traveltime_raster_path,
        bbox_gdf,
        projected_gdf.crs,
        target_friction_clipped_raster_path,
        reference_meta=ref_meta,
    )

    mask_array = rasterio.features.rasterize(
        ((geom, 1) for geom in projected_gdf.geometry),
        out_shape=(ref_meta["height"], ref_meta["width"]),
        transform=ref_meta["transform"],
        fill=0,
        dtype=rasterio.uint8,
    ).astype(np.int8)

    aoi_meta = ref_meta.copy()
    aoi_meta.update(
        {"count": 1, "dtype": rasterio.uint8, "nodata": 0, "compress": "lzw"}
    )

    with rasterio.open(target_aoi_raster_path, "w", **aoi_meta) as dst:
        dst.write(mask_array, 1)

    with rasterio.open(target_friction_clipped_raster_path) as friction_ds:
        friction_array = friction_ds.read(1)
        transform = friction_ds.transform
        cell_length_m = transform.a
        n_rows, n_cols = friction_array.shape

    travel_reach_array = shortest_distances.find_mask_reach(
        friction_array, mask_array, cell_length_m, n_cols, n_rows, max_time_mins
    )

    target_max_reach_raster_path = (
        working_dir / f"max_reach_{max_time_mins}min.tif"
    )

    with rasterio.open(
        target_max_reach_raster_path, "w", **aoi_meta
    ) as max_reach:
        max_reach.write(travel_reach_array, 1)

    pop_meta = ref_meta.copy()
    pop_meta.update(
        {"count": 1, "dtype": rasterio.int64, "nodata": 0, "compress": "lzw"}
    )
    pop_masked_array = np.where(
        (travel_reach_array > 0) & (pop_array > 0), pop_array, 0
    )
    with rasterio.open(target_pop_raster_path, "w", **pop_meta) as dst:
        dst.write(pop_masked_array, 1)
    return np.sum(pop_masked_array)


def calculate_ds_pop_from_conditional_raster(
    aoi_vector_path,
    flow_dir_raster_path,
    clipped_pop_raster_path,
    base_raster_path,
    condition_id,
    expression,
    buffer_size_m,
    wgs84_pixel_size,
    travel_time_pixel_size_m,
    working_dir,
    target_pop_raster_path,
):
    condition_raster_path = (
        working_dir / f"mask_{condition_id}_{base_raster_path.stem}.tif"
    )

    clipped_base_raster_path = (
        working_dir / f"clipped_{condition_id}_{base_raster_path.stem}.tif"
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
    )

    base_raster_nodata = geoprocessing.get_raster_info(base_raster_path)[
        "nodata"
    ][0]

    def _local_op(value):
        """Applies an elementwise expression to an array with nodata masking.

        Note: `base_raster_nodata` and `expression` are provided in the outside
            closure.

        Args:
            value (np.ndarray): Input array.

        Returns:
            np.ndarray: Array with the expression applied to valid elements.
        """
        result = value.copy()
        valid_mask = (
            slice(None)
            if base_raster_nodata is None
            else value != base_raster_nodata
        )
        result[valid_mask] = eval(
            expression,
            {"__builtins__": {}},
            {"value": value[valid_mask], "np": np},
        )
        return result

    geoprocessing.raster_calculator(
        [(str(clipped_base_raster_path), 1)],
        _local_op,
        condition_raster_path,
        gdal.GDT_Byte,
        None,
    )

    ds_coverage_raster_path = str(
        working_dir / f"ds_coverage_{condition_id}_{base_raster_path.stem}.tif"
    )

    routing.flow_accumulation_mfd(
        (str(flow_dir_raster_path), 1),
        str(ds_coverage_raster_path),
        weight_raster_path_band=(str(condition_raster_path), 1),
    )

    buffer_amounts_in_pixels = int(
        np.round(buffer_size_m / travel_time_pixel_size_m)
    )

    kernel_path = str(working_dir / f"{buffer_amounts_in_pixels}_kernel.tif")
    create_circular_kernel(kernel_path, buffer_amounts_in_pixels)
    buffered_ds_coverage_raster_path = "%s_buff%s" % os.path.splitext(
        str(ds_coverage_raster_path)
    )
    geoprocessing.convolve_2d(
        (ds_coverage_raster_path, 1),
        (kernel_path, 1),
        buffered_ds_coverage_raster_path,
        n_workers=1,
    )

    def mask_op(mask, pop_val):
        return np.where((mask > 0) & (pop_val > 0), pop_val, 0)

    pop_info = geoprocessing.get_raster_info(clipped_pop_raster_path)
    geoprocessing.raster_calculator(
        [
            (str(buffered_ds_coverage_raster_path), 1),
            (str(clipped_pop_raster_path), 1),
        ],
        mask_op,
        target_pop_raster_path,
        pop_info["datatype"],
        None,
    )
    return np.sum(gdal.OpenEx(target_pop_raster_path).ReadAsArray())


def calc_flow_dir(dem_path, working_dir, target_flow_dir_raster_path):
    pit_filled_raster_path = (
        working_dir / f"pit_filled_{Path(dem_path).stem}.tif"
    )
    routing.fill_pits(
        (dem_path, 1), pit_filled_raster_path, working_dir=working_dir
    )
    routing.flow_dir_mfd(
        (str(pit_filled_raster_path), 1),
        str(target_flow_dir_raster_path),
        working_dir=str(working_dir),
    )


def combine_pops(
    pop_raster_list,
    wgs84_pixel_size,
    working_dir,
    target_combined_pop_raster_path,
):
    def or_op(*value_list):
        result = value_list[0]
        for value_array in value_list[1:]:
            result |= value_array
        return result

    base_pop_raster_list = [str(path) for path in pop_raster_list]
    aligned_dir_path = working_dir / "aligned_pops"
    aligned_dir_path.mkdir(parents=True, exist_ok=True)
    aligned_pop_raster_list = [
        str(aligned_dir_path / os.path.basename(path))
        for path in base_pop_raster_list
    ]
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    wgs84_wkt = srs.ExportToWkt()
    geoprocessing.align_and_resize_raster_stack(
        base_pop_raster_list,
        aligned_pop_raster_list,
        ["near"] * len(aligned_pop_raster_list),
        [wgs84_pixel_size, -wgs84_pixel_size],
        "union",
        target_projection_wkt=wgs84_wkt,
    )
    geoprocessing.raster_calculator(
        [(str(path), 1) for path in aligned_pop_raster_list],
        or_op,
        target_combined_pop_raster_path,
        gdal.GDT_Float32,
        None,
    )
    return np.sum(gdal.OpenEx(target_combined_pop_raster_path).ReadAsArray())


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description="Extract and normalize analysis config from YAML."
    )
    ap.add_argument("config", type=Path, help="Path to YAML config file")
    ap.add_argument(
        "--json", action="store_true", help="Print normalized JSON to stdout"
    )
    ap.add_argument(
        "--validate-paths",
        action="store_true",
        help="Lightly validate paths exist (non-glob)",
    )
    args = ap.parse_args()

    config = process_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        config["logging"]["level"], config["logging"]["to_file"]
    )

    validate_paths(config)
    logger.info(f"{args.config} read successfully")

    aoi_id_to_path = collect_aoi_files(config)
    logger.info(f"found {len(aoi_id_to_path)} aois to process")

    n_workers = min(len(aoi_id_to_path), psutil.cpu_count(logical=False))
    update_rate = 15.0

    task_graph = taskgraph.TaskGraph(config["work_dir"], n_workers, update_rate)
    logging.getLogger("ecoshard").setLevel(config["logging"]["level"])
    logging.basicConfig()

    # these drive how we cut, reproject, and determine distance
    wgs84_pixel_size = config["inputs"]["wgs84_pixel_size"]

    good_crs_for_aoi = {}
    for aoi_key, aoi_vector_path in aoi_id_to_path.items():
        crs_task = task_graph.add_task(
            func=choose_equidistant_crs_from_bbox,
            args=(aoi_vector_path,),
            store_result=True,
            task_name=f"calculate CRS for {aoi_key}",
        )
        good_crs_for_aoi[aoi_key] = crs_task

    section_mask_ids = set()
    pop_results = collections.defaultdict(dict)
    for aoi_key, aoi_vector_path in aoi_id_to_path.items():
        working_dir = Path(config["work_dir"]) / Path(aoi_key)
        working_dir.mkdir(parents=True, exist_ok=True)
        target_subwatershed_path = working_dir / Path(
            f"downstream_watersheds_from_{aoi_key}.gpkg"
        )
        subset_task = task_graph.add_task(
            func=subset_subwatersheds,
            args=(
                aoi_vector_path,
                config["inputs"]["subwatershed_vector_path"],
                good_crs_for_aoi[aoi_key].get(),
                target_subwatershed_path,
            ),
            target_path_list=[target_subwatershed_path],
            task_name=f"subset downstream from {aoi_key}",
        )
        wgs84_pixel_size = config["inputs"]["wgs84_pixel_size"]
        base_raster_path_list = [
            config["inputs"]["population_raster_path"],
            config["inputs"]["dem_raster_path"],
        ]
        target_clipped_raster_path_list = [
            str(working_dir / f"{Path(path).stem}_clipped.tif")
            for path in base_raster_path_list
        ]

        clipped_pop_raster_path = target_clipped_raster_path_list[0]
        clipped_dem_path = target_clipped_raster_path_list[1]

        clip_task = task_graph.add_task(
            func=align_and_resize_raster_stack_on_vector,
            args=(
                base_raster_path_list,
                target_clipped_raster_path_list,
                ["near"] * len(base_raster_path_list),
                [wgs84_pixel_size, -wgs84_pixel_size],
                target_subwatershed_path,
            ),
            dependent_task_list=[subset_task],
            target_path_list=target_clipped_raster_path_list,
            task_name=f"clip base data for {aoi_key}",
        )

        target_flow_dir_raster_path = "%s_mfdflow%s" % os.path.splitext(
            str(clipped_dem_path)
        )
        flow_dir_task = task_graph.add_task(
            func=calc_flow_dir,
            args=(clipped_dem_path, working_dir, target_flow_dir_raster_path),
            dependent_task_list=[clip_task],
            target_path_list=[target_flow_dir_raster_path],
            task_name=f"calculate flow dir for {aoi_key}",
        )

        pop_rasters = []
        pop_raster_tasks = []
        for mask_section in config["masks"]:
            section_id = mask_section["id"]
            section_mask_ids.add(section_id)
            target_pop_raster_path = (
                output_dir / f"{aoi_key}_{mask_section['id']}_pop.tif"
            )
            pop_rasters.append(target_pop_raster_path)
            if mask_section["type"] == "travel_time_population":
                travel_task = task_graph.add_task(
                    func=apply_travel_time_mask,
                    args=(
                        config["inputs"]["traveltime_raster_path"],
                        config["inputs"]["population_raster_path"],
                        aoi_vector_path,
                        mask_section["params"]["max_hours"],
                        good_crs_for_aoi[aoi_key].get().crs,
                        target_pop_raster_path,
                        working_dir,
                    ),
                    store_result=True,
                    target_path_list=[target_pop_raster_path],
                    task_name=f"travel time for {aoi_key}",
                )
                pop_results[aoi_key][section_id] = travel_task
                pop_raster_tasks.append(travel_task)
            elif mask_section["type"] == "conditional_raster":
                conditional_task = task_graph.add_task(
                    func=calculate_ds_pop_from_conditional_raster,
                    args=(
                        aoi_vector_path,
                        Path(target_flow_dir_raster_path),
                        Path(clipped_pop_raster_path),
                        Path(mask_section["params"]["condition_raster_path"]),
                        section_id,
                        mask_section["params"]["expression"],
                        config["inputs"]["buffer_size_m"],
                        config["inputs"]["wgs84_pixel_size"],
                        config["inputs"]["travel_time_pixel_size_m"],
                        working_dir,
                        target_pop_raster_path,
                    ),
                    dependent_task_list=[flow_dir_task],
                    store_result=True,
                    target_path_list=[target_pop_raster_path],
                    task_name=f"conditional downstream {section_id}",
                )
                pop_results[aoi_key][section_id] = conditional_task
                pop_raster_tasks.append(conditional_task)
            else:
                raise ValueError(
                    f"unknown mask section type: {mask_section['type']}"
                )
        target_combined_pop_raster_path = (
            output_dir / f"{aoi_key}_total_pop.tif"
        )
        task_graph.join()
        target_combined_pop_raster_path = (
            output_dir / f"{aoi_key}_total_pop.tif"
        )
        combined_task = task_graph.add_task(
            func=combine_pops,
            args=(
                pop_rasters,
                wgs84_pixel_size,
                working_dir,
                target_combined_pop_raster_path,
            ),
            dependent_task_list=pop_raster_tasks,
            store_result=True,
            target_path_list=[target_combined_pop_raster_path],
            task_name=f"combined pop for {aoi_key}",
        )
        combined_header = "combined pop"
        section_mask_ids.add(combined_header)
        pop_results[aoi_key][combined_header] = combined_task

    task_graph.join()
    rows = []
    for aoi_key, results in pop_results.items():
        row = {"aoi": aoi_key}
        for header in section_mask_ids:
            row[header] = results.get(header, "n/a").get()
        rows.append(row)

    df = pd.DataFrame(rows, columns=["aoi"] + list(section_mask_ids))
    cols = (
        ["aoi"]
        + [c for c in df.columns if c not in ("aoi", "combined pop")]
        + ["combined pop"]
    )
    df = df[cols]
    csv_path = (
        output_dir
        / f'{config["run_name"]}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'
    )
    df.to_csv(csv_path, index=False)
    task_graph.close()


if __name__ == "__main__":
    main()
