"""Runner for gathering ds pop info given masks.

To run in docker environment:
- windows:
docker build -t therealspring/ds_beneficiaries:latest . && docker run --rm -it -v "%CD%":/usr/local/wwf_es_beneficiaries therealspring/ds_beneficiaries:latest
- linux/mac:
docker build -t ds_beneficiaries:latest . && docker run --rm -it -v `pwd`:/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import glob
import logging
import sys
import time
from itertools import islice

from ecoshard import taskgraph
import psutil
import yaml

from ecoshard import geoprocessing
from ecoshard import taskgraph
from ecoshard.geoprocessing import routing
from osgeo import gdal
from shapely.geometry import box
from shapely.geometry import shape
from shapely.ops import transform
import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import rasterio
import pyogrio
from pyproj import CRS, Transformer


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
    aoi_vector_pattern = _as_list(inputs.get("aoi_vector_pattern", []))

    missing_messages = []
    if not population_raster_path:
        missing_messages.append(
            "population_raster_path (path to population raster)"
        )

    if not traveltime_raster_path:
        missing_messages.append(
            "traveltime_raster_path (path to travel-time raster)"
        )

    if not subwatershed_vector_path:
        missing_messages.append(
            "subwatershed_vector_path (path to subwatershed shapefile/vector)"
        )

    if not aoi_vector_pattern:
        missing_messages.append(
            "aoi_vector_pattern (one or more AOI file patterns)"
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
            "population_raster_path": population_raster_path,
            "traveltime_raster_path": traveltime_raster_path,
            "subwatershed_vector_path": subwatershed_vector_path,
            "aoi_vector_pattern": aoi_vector_pattern,
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
    logger = logging.getLogger(__name__)
    logger.setLevel(level.upper())
    logger.handlers.clear()

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    format_str = (
        "%(asctime)s %(filename)s:%(lineno)d [%(levelname)s]  %(message)s"
    )
    sh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)

    return logger


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
    subset_subwatersheds_vector_path: str | Path,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"processing aoi {aoi_vector_path}")

    # Read AOI and precompute union + bbox; repair invalid with make_valid if available
    aoi_gdf = gpd.read_file(aoi_vector_path)
    aoi_crs = aoi_gdf.crs
    if hasattr(aoi_gdf.geometry, "make_valid"):
        aoi_gdf = aoi_gdf.set_geometry(aoi_gdf.geometry.make_valid())
    else:
        aoi_gdf = aoi_gdf.set_geometry(aoi_gdf.geometry.buffer(0))
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
        gpd.GeoDataFrame(geometry=[], crs=aoi_crs).to_file(
            subset_subwatersheds_vector_path, driver="GPKG"
        )
        logger.warning("No candidates found in bbox; wrote empty GPKG.")
        return

    hits = sub_bbox_gdf.sindex.query(aoi_union, predicate="intersects")
    if len(hits) == 0:
        gpd.GeoDataFrame(geometry=[], crs=aoi_crs).to_file(
            subset_subwatersheds_vector_path, driver="GPKG"
        )
        logger.warning("No intersecting sub-watersheds; wrote empty GPKG.")
        return

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
        gpd.GeoDataFrame(geometry=[], crs=aoi_crs).to_file(
            subset_subwatersheds_vector_path, driver="GPKG"
        )
        logger.warning("No valid geometry found; wrote empty GPKG.")
        return

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
        gpd.GeoDataFrame(geometry=[], crs=aoi_crs).to_file(
            subset_subwatersheds_vector_path, driver="GPKG"
        )
        logger.warning("No geometries returned; wrote empty GPKG.")
        return

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

    sub_gdf.to_file(subset_subwatersheds_vector_path, driver="GPKG")
    logger.info(f"all done subwatershedding {aoi_vector_path}")


# def subset_subwatersheds(
#     aoi_vector_path,
#     subwatershed_vector_path,
#     target_subset_subwatersheds_vector_path,
# ):
#     logger = logging.getLogger(__name__)
#     logger.info(f"processing aoi {aoi_vector_path}")
#     aoi_gdf = gpd.read_file(aoi_vector_path)
#     aoi_gdf["geometry"] = aoi_gdf["geometry"].buffer(0)
#     aoi_union = aoi_gdf.unary_union
#     aoi_bbox = box(*aoi_gdf.total_bounds)
#     aoi_crs = aoi_gdf.crs

#     logger.info(f"Sub‑watershed CRS & ID ->NEXT mapping  {aoi_vector_path}")
#     with fiona.open(subwatershed_vector_path, "r") as src:
#         sub_crs = src.crs
#         layer_name = src.name
#         hybas_to_nextdown = {
#             f["properties"]["HYBAS_ID"]: f["properties"]["NEXT_DOWN"]
#             for f in src
#         }

#     logger.info(f"Reproject AOI to sub‑watershed CRS {aoi_vector_path}")
#     if aoi_crs != sub_crs:
#         tform = pyproj.Transformer.from_crs(
#             aoi_crs, sub_crs, always_xy=True
#         ).transform
#         aoi_bbox = transform(tform, aoi_bbox)
#         aoi_union = transform(tform, aoi_union)

#     logger.info(f"Fast spatial pre-filter {aoi_vector_path}")
#     sub_bbox_gdf = gpd.read_file(subwatershed_vector_path, bbox=aoi_bbox.bounds)
#     hits = sub_bbox_gdf.sindex.query(aoi_union, predicate="intersects")
#     initial = sub_bbox_gdf.iloc[hits]

#     logger.info(f"Downstream BFS {aoi_vector_path}")
#     all_ids = set(initial["HYBAS_ID"])
#     queue = set(initial["NEXT_DOWN"]) - {0}
#     while queue:
#         all_ids.update(queue)
#         queue = (
#             {
#                 hybas_to_nextdown.get(hid)
#                 for hid in queue
#                 if hid in hybas_to_nextdown
#             }
#             - all_ids
#             - {0}
#         )

#     if not all_ids:
#         gpd.GeoDataFrame({"geometry": []}, crs=aoi_crs).to_file(
#             target_subset_subwatersheds_vector_path, driver="GPKG"
#         )
#         logger.warning("No valid geometry found; wrote empty GPKG.")
#         return

#     logger.info(f"Fetch geometries with attribute SQL {aoi_vector_path}")
#     gdf_parts = []
#     for id_chunk in _chunks(all_ids, 1000):
#         id_list = ",".join(map(str, id_chunk))
#         sql = f'SELECT * FROM "{layer_name}" WHERE HYBAS_ID IN ({id_list})'
#         gdf_parts.append(gpd.read_file(subwatershed_vector_path, sql=sql))
#     sub_gdf = gpd.GeoDataFrame(
#         pd.concat(gdf_parts, ignore_index=True), crs=sub_crs
#     )

#     logger.info(f"Repair only invalid geometries {aoi_vector_path}")
#     invalid = ~sub_gdf.geometry.is_valid
#     if invalid.any():
#         sub_gdf.loc[invalid, "geometry"] = sub_gdf.loc[
#             invalid, "geometry"
#         ].buffer(0)

#     logger.info(f"Reproject to AOI CRS & write {aoi_vector_path}")
#     if sub_crs != aoi_crs:
#         sub_gdf = sub_gdf.to_crs(aoi_crs)
#     sub_gdf.to_file(target_subset_subwatersheds_vector_path, driver="GPKG")
#     logger.info(f"all done subwatershedding {aoi_vector_path}")


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
    logger = setup_logger(
        config["logging"]["level"], config["logging"]["to_file"]
    )

    validate_paths(config)
    logger.info(f"{args.config} read successfully")

    aoi_id_to_path = collect_aoi_files(config)
    logger.debug(aoi_id_to_path)
    logger.info(f"found {len(aoi_id_to_path)} aois to process")

    n_workers = min(len(aoi_id_to_path), psutil.cpu_count(logical=False))
    update_rate = 15.0
    logging.basicConfig()
    task_graph = taskgraph.TaskGraph(config["work_dir"], n_workers, update_rate)
    logging.getLogger("ecoshard").setLevel(config["logging"]["level"])
    for aoi_key, aoi_path in aoi_id_to_path.items():
        working_dir = Path(config["work_dir"]) / Path(aoi_key)
        working_dir.mkdir(parents=True, exist_ok=True)
        target_subwatershed_path = working_dir / Path(
            f"downstream_watersheds_from_{aoi_key}.gpkg"
        )
        subset_task = task_graph.add_task(
            func=subset_subwatersheds,
            args=(
                aoi_path,
                config["inputs"]["subwatershed_vector_path"],
                target_subwatershed_path,
            ),
            target_path_list=[target_subwatershed_path],
            task_name=f"subset downstream from {aoi_key}",
        )

    task_graph.join()
    task_graph.close()

    # TODO:
    # - loop through each aoi vector pattern
    #   - create a working directory for the project under config["work_dir"]/aoi.stem
    #   - determine which features in subwatershed_vector_path intersect with
    #     the aoi
    #     - make a set of the features which intersect and ALL features that
    #       are downstream of it until there are no more downstream. The
    #       downstream feature is identified by looking at the current feature's
    #       `NEXT_DOWN` field. If it is 0 there are no downstream features
    #       but if it is non-zero then it will match with a feature that has
    #       the same `HYBAS_ID`
    #     - find an appropriate equadistant projection for this subset (*TODO)
    #   - clip DEM by that subset in wgs84 then project into the subset
    #     - route the DEM
    #   - loop through each section
    #     - clip the condition raster path or the travel time raster to the
    #       subset of downstream and reproject the result into the appropriate
    #       equadistant projection
    #     - either:
    #       - apply the expression to the conditional raster
    #         - do a flow accumulation of the result of the conditional raster
    #         - (?) buffer it out a given set of meters?
    #       - calculate the travel time to the rasterized AOI
    #         - mask the travel time <= `max_hours` parameter
    #     - apply the raster calcualted in the previous section to the
    #       population by setting population = 0 where the mask is = 0 (this is
    #       a result, put it in `output_dir` called population_`id`_
    #       {aoi.stem}).tif keep a reference for later
    #       - sum the raster and keep in memory for later
    #   - make one final population raster that is a logical OR of the masked
    #     population layers calculated earlier
    #     - sum it and keep in memory for later
    #   - generate a table that has columns called 'aoi', then as many columns
    #     are in the -masks sections named after the `id` in them suffixed by
    #     ds_pop, then one more column called combined_ds_pop, save that in the
    #     output_dir called `run_name`_analysis_YYYY_MM_DD_MM_HH_SS.csv`


if __name__ == "__main__":
    main()
