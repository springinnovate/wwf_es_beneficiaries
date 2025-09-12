"""Runner for gathering ds pop info given masks.

To run in docker environment:
- windows:
docker build -t ds_beneficiaries:latest . && docker run --rm -it -v "%CD%":/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest
- linux/mac:
docker build -t ds_beneficiaries:latest . && docker run --rm -it -v `pwd`:/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest
"""

import argparse
import json
from pathlib import Path
import logging
from typing import Any, Dict, List, Tuple
import sys

import yaml


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def process_config(config_path: Path) -> Dict[str, Any]:
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

    # check if any of the core inputs are missing
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
        log_file (str): Path to the log file. If empty, no file handler is added.

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


def main() -> None:
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
    # print_yaml_config(config)


if __name__ == "__main__":
    main()
