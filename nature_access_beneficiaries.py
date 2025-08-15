"""Distance to habitat with a friction layer."""

from datetime import datetime
import glob
import logging
import os
import sys

from ecoshard import taskgraph
from osgeo import gdal
from pyproj import CRS
from pyproj import Transformer
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from shapely.geometry import shape
import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rasterio
import rasterio.mask

import shortest_distances

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__name__)
logging.getLogger("ecoshard.taskgraph").setLevel(logging.INFO)
logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARNING)


AOI_DIRS = [
    "./data/aois/aoi_by_country",
    "./data/aois/WWF-Int_Pilot-sites",
]

ANALYSIS_AOIS = {}
BAD_AOIS = {}

for aoi_dir in AOI_DIRS:
    for ext in ["shp", "gpkg"]:
        pattern = os.path.join(aoi_dir, "**", f"*.{ext}")
        for file_path in glob.glob(pattern, recursive=True):
            basename = os.path.splitext(os.path.basename(file_path))[0]
            try:
                with fiona.open(file_path, "r") as src:
                    if len(src) == 0:
                        raise ValueError("No features found")

                    valid_geometry_found = False
                    for feature in src:
                        geom = feature["geometry"]
                        if geom and geom != {}:
                            shapely_geom = shape(geom)
                            if not shapely_geom.is_empty:
                                valid_geometry_found = True
                                break

                    if not valid_geometry_found:
                        raise ValueError("No valid geometry found in features")

                ANALYSIS_AOIS[basename] = file_path
            except Exception:
                BAD_AOIS[basename] = file_path


FRICTION_SURFACE_RASTER_PATH = "./data/travel_time/friction_surface_2019_compressed_md5_1be7dd230178a5d395529be7a5e3fb0a.tif"
POPULATION_RASTER_PATH = "./data/pop_rasters/landscan-global-2023.tif"
WORKSPACE_DIR = "workspace_nature_access_analysis"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

TRAVEL_TIME_HOURS = list(range(1, 2))


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


def get_utm_crs(geometry, original_crs):
    LOGGER.info("getting the utm crs")

    if not original_crs.is_geographic:
        LOGGER.info("not in lat/lon, reprojecting for that")
        geometry = (
            gpd.GeoSeries([geometry], crs=original_crs).to_crs("EPSG:4326").iloc[0]
        )

    bounds = geometry.bounds
    width = abs(bounds[2] - bounds[0])
    height = abs(bounds[3] - bounds[1])

    if width > 6 or height > 6:
        LOGGER.info("Geometry exceeds 6 degrees, defaulting to Eckert IV projection")
        return CRS.from_proj4("+proj=eck4 +datum=WGS84 +units=m +no_defs")

    centroid = geometry.centroid
    utm_zone = int((centroid.x + 180) // 6) + 1
    hemisphere = "north" if centroid.y >= 0 else "south"

    LOGGER.info(f"determined utm_zone is : {utm_zone}")

    return CRS.from_dict(
        {"proj": "utm", "zone": utm_zone, "south": hemisphere == "south"}
    )


def clip_and_reproject_raster(
    base_raster_path, bbox_gdf, dst_crs, target_raster_path, reference_meta=None
):
    SPHERE_ECK4 = "+proj=eck4 +R=6371000 +units=m +no_defs"
    bbox_sph = bbox_gdf.set_crs(
        SPHERE_ECK4, allow_override=True
    )  # keep coords the same, just swap CRS tag
    bbox_wgs = bbox_sph.to_crs("EPSG:4326")  # should be finite
    LOGGER.info(f"********* quick test: {bbox_wgs.total_bounds}")

    with rasterio.open(base_raster_path) as src:
        if bbox_gdf.crs is None:
            raise ValueError("bbox_gdf must have a CRS defined")

        if "+proj=eck4" in bbox_gdf.crs.to_proj4().lower():
            LOGGER.info("eckert is so broken, just doing regular lat/lng bounds")
            projected_box_gdf = transform_edge_points_eckert_to_wgs84(bbox_gdf)
            projected_box_gdf = gpd.GeoDataFrame(
                geometry=[box(-179, -80, 179, 80)], crs="EPSG:4326"
            )
        else:
            # Check intermediate clamped bounds explicitly:
            LOGGER.info(f"Clamped bbox bounds: {bbox_gdf.total_bounds}")

            # Safely project to src.crs (e.g., EPSG:4326)
            projected_box_gdf = bbox_gdf.to_crs(src.crs)

        # Check bounds again after projection
        LOGGER.info(
            f"Projected bbox bounds: {projected_box_gdf.total_bounds} / {projected_box_gdf.crs.to_proj4()}"
        )
        bbox_projected_geom = [projected_box_gdf.geometry.iloc[0]]

        LOGGER.info(
            f"Original geom {bbox_gdf} {bbox_gdf.crs} converted to {src.crs}: {bbox_projected_geom}"
        )
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


def process_aoi(aoi_basename, aoi_path, max_hours, workspace_dir):
    max_time_mins = max_hours * 60
    local_workspace_dir = os.path.join(workspace_dir, aoi_basename)
    os.makedirs(local_workspace_dir, exist_ok=True)
    LOGGER.info(f"Processing AOI: {aoi_basename}")

    LOGGER.info(f"{aoi_basename}: Reading AOI shapefile from {aoi_path}")
    gdf = gpd.read_file(aoi_path)

    target_crs = get_utm_crs(gdf.union_all(), gdf.crs)
    LOGGER.info(
        f"{aoi_basename}: Projecting AOI geometry to appropriate UTM ({target_crs})"
    )
    projected_gdf = gdf.to_crs(target_crs)

    bbox = projected_gdf.total_bounds
    buffer_distance_m = max_hours * 104 * 1000  # drive 65mph for that many hours

    buffered_bbox = box(
        bbox[0] - buffer_distance_m,
        bbox[1] - buffer_distance_m,
        bbox[2] + buffer_distance_m,
        bbox[3] + buffer_distance_m,
    )

    LOGGER.info(f"buffered box: {buffered_bbox}")
    bbox_gdf = gpd.GeoDataFrame({"geometry": [buffered_bbox]}, crs=projected_gdf.crs)

    target_pop_clipped_raster_path = os.path.join(
        local_workspace_dir,
        f"{aoi_basename}_population_clipped_{max_time_mins}min.tif",
    )
    target_friction_clipped_raster_path = os.path.join(
        local_workspace_dir,
        f"{aoi_basename}_friction_surface_clipped_{max_time_mins}min.tif",
    )
    target_aoi_raster_path = os.path.join(
        local_workspace_dir,
        f"{aoi_basename}_aoi_rasterized_{max_time_mins}min.tif",
    )

    LOGGER.info(f"{aoi_basename}: Clipping and reprojecting population raster")
    clip_and_reproject_raster(
        POPULATION_RASTER_PATH,
        bbox_gdf,
        projected_gdf.crs,
        target_pop_clipped_raster_path,
    )

    LOGGER.info(f"{aoi_basename}: Reading clipped population raster")
    with rasterio.open(target_pop_clipped_raster_path) as pop_ref:
        ref_meta = pop_ref.meta.copy()
        pop_array = pop_ref.read(1).astype(np.int64)

    LOGGER.info(f"{aoi_basename}: Clipping and reprojecting friction surface raster")
    clip_and_reproject_raster(
        FRICTION_SURFACE_RASTER_PATH,
        bbox_gdf,
        projected_gdf.crs,
        target_friction_clipped_raster_path,
        reference_meta=ref_meta,
    )

    LOGGER.info(f"{aoi_basename}: Rasterizing AOI geometry")
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

    LOGGER.info(f"{aoi_basename}: Writing rasterized AOI to {target_aoi_raster_path}")
    with rasterio.open(target_aoi_raster_path, "w", **aoi_meta) as dst:
        dst.write(mask_array, 1)

    LOGGER.info(f"{aoi_basename}: Reading friction surface raster")
    with rasterio.open(target_friction_clipped_raster_path) as friction_ds:
        friction_array = friction_ds.read(1)
        transform = friction_ds.transform
        cell_length_m = transform.a
        n_rows, n_cols = friction_array.shape

    LOGGER.info(
        f"{aoi_basename}: Calculating travel reach raster ({n_cols} cols x {n_rows} rows)"
    )
    target_max_reach_raster_path = os.path.join(
        local_workspace_dir, f"{aoi_basename}_max_reach_{max_time_mins}min.tif"
    )

    travel_reach_array = shortest_distances.find_mask_reach(
        friction_array, mask_array, cell_length_m, n_cols, n_rows, max_time_mins
    )

    LOGGER.info(
        f"{aoi_basename}: Writing travel reach raster to {target_max_reach_raster_path}"
    )
    with rasterio.open(target_max_reach_raster_path, "w", **aoi_meta) as max_reach:
        max_reach.write(travel_reach_array, 1)

    in_range_pop_count = np.sum(pop_array[(travel_reach_array > 0) & (pop_array > 0)])

    LOGGER.info(
        f"{aoi_basename}: Population within {max_hours} hours travel: {in_range_pop_count}"
    )
    return in_range_pop_count


def main():
    """Entry point."""
    n_workers = psutil.cpu_count(logical=False)
    taskgraph_update_interval_s = 15.0
    task_graph = taskgraph.TaskGraph(
        WORKSPACE_DIR, n_workers, taskgraph_update_interval_s
    )
    results_dict = {}
    task_list = []
    for aoi_basename, aoi_path in ANALYSIS_AOIS.items():
        for max_hours in TRAVEL_TIME_HOURS:
            in_range_pop_count_task = task_graph.add_task(
                func=process_aoi,
                args=(aoi_basename, aoi_path, max_hours, WORKSPACE_DIR),
                store_result=True,
                transient_run=True,
                task_name=f"process {aoi_basename} at {max_hours} hours",
            )
            task_list.append((aoi_basename, max_hours, in_range_pop_count_task))
    for aoi_basename, max_hours, in_range_pop_count_task in task_list:
        if aoi_basename not in results_dict:
            results_dict[aoi_basename] = {}
        results_dict[aoi_basename][max_hours] = in_range_pop_count_task.get()

    df = pd.DataFrame(results_dict).T.sort_index(axis=0)
    df = df.sort_index(axis=1)

    # Rename columns clearly
    df.columns = [f"{hour}_hours" for hour in df.columns]

    # Save DataFrame to CSV with current timestamp in filename
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_filename = f"people_within_travel_time_{timestamp}.csv"
    df.to_csv(csv_filename)

    LOGGER.info(f"Saved results to {csv_filename}")


if __name__ == "__main__":
    main()
