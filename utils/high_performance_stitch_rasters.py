"""Stitch rasters listed in a text file into one common output grid.

This script is intentionally standalone: it does not import any project modules
from this repository. It uses the first input raster as the reference for CRS,
pixel size, dtype, band count, and nodata. The output extent is the union of all
input raster bounds, aligned to the first raster's pixel grid.

Example:
    python utils/high_performance_stitch_rasters.py rasters.txt stitched.tif

The raster list should contain one path per line. Blank lines and lines whose
first non-whitespace character is ``#`` are ignored. Relative paths are resolved
relative to the list file.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected but not required.
    tqdm = None


DEFAULT_BLOCK_SIZE = 256
DEFAULT_CREATION_OPTIONS = {
    "BIGTIFF": "YES",
    "NUM_THREADS": "ALL_CPUS",
    "compress": "lzw",
    "tiled": True,
    "blockxsize": DEFAULT_BLOCK_SIZE,
    "blockysize": DEFAULT_BLOCK_SIZE,
}


def _progress(iterable: Iterable, **kwargs) -> Iterable:
    """Wrap ``iterable`` in tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def read_raster_list(list_path: Path) -> list[Path]:
    """Read raster paths from ``list_path``.

    Args:
        list_path: Text file with one raster path per line. Blank lines and
            comment lines starting with ``#`` are ignored. Relative raster paths
            are resolved relative to ``list_path.parent``.

    Returns:
        Ordered list of raster paths.

    Raises:
        ValueError: If no raster paths are found.
        FileNotFoundError: If a listed raster does not exist.
    """
    list_path = Path(list_path).resolve()
    raster_paths: list[Path] = []
    with list_path.open("r", encoding="utf-8") as path_file:
        for line_number, raw_line in enumerate(path_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            raster_path = Path(line)
            if not raster_path.is_absolute():
                raster_path = list_path.parent / raster_path
            raster_path = raster_path.resolve()
            if not raster_path.exists():
                raise FileNotFoundError(
                    f"Raster listed on line {line_number} does not exist: "
                    f"{raster_path}"
                )
            raster_paths.append(raster_path)

    if not raster_paths:
        raise ValueError(f"No raster paths were found in {list_path}")
    return raster_paths


def _require_north_up(transform: Affine, raster_path: Path) -> None:
    """Validate that a raster transform is north-up.

    Args:
        transform: Raster affine transform to validate.
        raster_path: Source raster path used in error messages.

    Raises:
        ValueError: If the transform is rotated, flipped, or otherwise not
            north-up with positive x pixel size and negative y pixel size.
    """
    if not math.isclose(transform.b, 0.0) or not math.isclose(transform.d, 0.0):
        raise ValueError(f"Rotated rasters are not supported: {raster_path}")
    if transform.a <= 0 or transform.e >= 0:
        raise ValueError(
            "Expected a north-up raster with positive x pixel size and "
            f"negative y pixel size: {raster_path}"
        )


def _aligned_output_grid(
    raster_paths: Sequence[Path],
) -> tuple[dict, tuple[float, float, float, float]]:
    """Build an output Rasterio profile from input rasters."""
    with rasterio.open(raster_paths[0]) as reference:
        _require_north_up(reference.transform, raster_paths[0])
        reference_crs = reference.crs
        if reference_crs is None:
            raise ValueError(f"Reference raster has no CRS: {raster_paths[0]}")
        reference_transform = reference.transform
        pixel_width = reference_transform.a
        pixel_height = abs(reference_transform.e)
        origin_x = reference_transform.c
        origin_y = reference_transform.f
        profile = reference.profile.copy()
        reference_count = reference.count
        reference_dtypes = reference.dtypes

    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf

    for raster_path in raster_paths:
        with rasterio.open(raster_path) as source:
            _require_north_up(source.transform, raster_path)
            if source.crs != reference_crs:
                raise ValueError(
                    f"CRS mismatch for {raster_path}: expected "
                    f"{reference_crs}, got {source.crs}"
                )
            if source.count != reference_count:
                raise ValueError(
                    f"Band-count mismatch for {raster_path}: expected "
                    f"{reference_count}, got {source.count}"
                )
            if source.dtypes != reference_dtypes:
                raise ValueError(
                    f"Dtype mismatch for {raster_path}: expected "
                    f"{reference_dtypes}, got {source.dtypes}"
                )
            minx = min(minx, source.bounds.left)
            miny = min(miny, source.bounds.bottom)
            maxx = max(maxx, source.bounds.right)
            maxy = max(maxy, source.bounds.top)

    left = origin_x + math.floor((minx - origin_x) / pixel_width) * pixel_width
    right = origin_x + math.ceil((maxx - origin_x) / pixel_width) * pixel_width
    top = origin_y - math.floor((origin_y - maxy) / pixel_height) * pixel_height
    bottom = origin_y - math.ceil((origin_y - miny) / pixel_height) * pixel_height

    width = int(round((right - left) / pixel_width))
    height = int(round((top - bottom) / pixel_height))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Combined raster dimensions are invalid: {width}x{height}"
        )

    profile.update(
        {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "transform": from_origin(left, top, pixel_width, pixel_height),
        }
    )
    profile.update(DEFAULT_CREATION_OPTIONS)
    return profile, (left, bottom, right, top)


def _integer_window(raw_window: Window, width: int, height: int) -> Window:
    """Round ``raw_window`` outward and clamp it to raster dimensions."""
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


def _iter_block_windows(
    window: Window,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Iterator[Window]:
    """Yield block-sized windows within ``window``."""
    row_start = int(window.row_off)
    col_start = int(window.col_off)
    row_stop = int(window.row_off + window.height)
    col_stop = int(window.col_off + window.width)
    for row_off in range(row_start, row_stop, block_size):
        block_height = min(block_size, row_stop - row_off)
        for col_off in range(col_start, col_stop, block_size):
            block_width = min(block_size, col_stop - col_off)
            yield Window(col_off, row_off, block_width, block_height)


def _valid_mask(data: np.ndarray, nodata) -> np.ndarray:
    """Return a mask where ``data`` is not nodata."""
    if nodata is None:
        return np.ones(data.shape, dtype=bool)
    if np.issubdtype(data.dtype, np.floating) and np.isnan(nodata):
        return ~np.isnan(data)
    return data != nodata


def _initialize_output(target: rasterio.DatasetWriter, nodata) -> None:
    """Fill the output raster with nodata when nodata is defined."""
    if nodata is None:
        return

    for _, window in _progress(
        target.block_windows(1),
        desc="initialize output",
        unit="block",
    ):
        shape = (target.count, int(window.height), int(window.width))
        fill_block = np.full(shape, nodata, dtype=target.dtypes[0])
        target.write(fill_block, window=window)


def stitch_rasters(
    raster_paths: Sequence[Path],
    output_path: Path,
) -> Path:
    """Stitch ``raster_paths`` into ``output_path``.

    Later rasters overwrite earlier rasters wherever the later raster has valid
    data. Source nodata pixels are skipped.
    """
    if not raster_paths:
        raise ValueError("At least one raster path is required.")

    raster_paths = [Path(path).resolve() for path in raster_paths]
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_profile, _ = _aligned_output_grid(raster_paths)
    output_nodata = output_profile.get("nodata")

    with rasterio.open(output_path, "w", **output_profile) as output:
        _initialize_output(output, output_nodata)

    with rasterio.open(output_path, "r+") as output:
        for raster_path in _progress(
            raster_paths,
            desc="stitch rasters",
            unit="raster",
        ):
            with rasterio.open(raster_path) as source:
                target_window = _integer_window(
                    from_bounds(*source.bounds, transform=output.transform),
                    output.width,
                    output.height,
                )
                if target_window.width == 0 or target_window.height == 0:
                    continue

                vrt_kwargs = {
                    "crs": output.crs,
                    "transform": output.transform,
                    "width": output.width,
                    "height": output.height,
                    "resampling": Resampling.nearest,
                }
                if output_nodata is not None:
                    vrt_kwargs["nodata"] = output_nodata
                if source.nodata is not None:
                    vrt_kwargs["src_nodata"] = source.nodata

                with WarpedVRT(source, **vrt_kwargs) as source_vrt:
                    effective_nodata = (
                        source_vrt.nodata
                        if source_vrt.nodata is not None
                        else source.nodata
                    )
                    for window in _iter_block_windows(target_window):
                        data = source_vrt.read(window=window)
                        valid = _valid_mask(data, effective_nodata)
                        if not np.any(valid):
                            continue
                        if np.all(valid):
                            output.write(data, window=window)
                            continue
                        existing = output.read(window=window)
                        existing[valid] = data[valid]
                        output.write(existing, window=window)

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Stitch rasters listed in a text file into a common output grid. "
            "The first raster defines CRS, pixel size, dtype, band count, and "
            "nodata. Later rasters overwrite earlier rasters where they have "
            "valid data."
        )
    )
    parser.add_argument(
        "raster_list",
        type=Path,
        help="Text file with one raster path per line.",
    )
    parser.add_argument(
        "output_raster",
        type=Path,
        help="Output GeoTIFF path.",
    )
    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()
    raster_paths = read_raster_list(args.raster_list)
    output_path = stitch_rasters(raster_paths, args.output_raster)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
