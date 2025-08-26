"""Distance to habitat with a friction layer."""

from datetime import datetime
import logging
import os
import sys

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from pyproj import CRS
from pyproj import Transformer
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rasterio
import shortest_distances
from rasterio.transform import from_bounds

GTIFF_CREATION_TUPLE_OPTIONS = (
    "GTIFF",
    (
        "TILED=YES",
        "BIGTIFF=YES",
        "COMPRESS=LZW",
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
    ),
)

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


MASK_RASTER_PATH_DICT = {
    "A_25": os.path.join(
        "workspace_clip_and_analyze_CNA",
        "A_25_masked_A_25_md5_737a2625eda959bc0e9c024919e5e7b9.tif",
    ),
    "A_50": os.path.join(
        "workspace_clip_and_analyze_CNA",
        "A_50_masked_A_50_md5_e26abccbc56f1188e269458fe427073f.tif",
    ),
    "A_90": os.path.join(
        "workspace_clip_and_analyze_CNA",
        "A_90_masked_A_90_md5_79f5e0d5d5029d90e8f10d5932da93ff.tif",
    ),
}

FRICTION_SURFACE_RASTER_PATH = "./data/travel_time/friction_surface_2019_compressed_md5_1be7dd230178a5d395529be7a5e3fb0a.tif"
POPULATION_RASTER_PATH = "./data/pop_rasters/landscan-global-2023.tif"
WORKSPACE_DIR = "workspace_target_setting_dist_to_hab_with_friction"
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


def transform_bbox(bbox_geom, src_wkt, dst_wkt):
    transformer = Transformer.from_crs(src_wkt, dst_wkt, always_xy=True)
    return shapely_transform(
        lambda x, y, z=None: transformer.transform(x, y), bbox_geom
    )


# def clip_raster(src_path, bbox_geom, dst_path):
def reproject_and_pad_raster(
    src_path, bbox_geom, dst_path, resolution=None, dst_crs=None, nodata=None
):
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        dst_crs = dst_crs if dst_crs else src_crs

        minx, miny, maxx, maxy = bbox_geom.bounds

        # Set resolution to source resolution if not specified
        resolution = resolution if resolution else (src.res[0], src.res[1])

        # Calculate dimensions explicitly
        width = int((maxx - minx) / resolution[0])
        height = int((maxy - miny) / resolution[1])

        dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Prepare metadata for the destination raster
        dst_meta = src.meta.copy()
        dst_meta.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "nodata": nodata if nodata is not None else src.nodata,
            }
        )

        # Create destination raster and explicitly reproject/pad
        with rasterio.open(dst_path, "w", **dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=dst_meta["nodata"],
                )


def reproject_raster(src_path, dst_path, dst_crs, dst_res):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=dst_res,
        )
        meta = src.meta.copy()
        meta.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )
        with rasterio.open(dst_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def process_raster_mask(mask_id, mask_raster_path, max_hours, workspace_dir):
    max_time_mins = max_hours * 60
    local_workspace_dir = os.path.join(workspace_dir, mask_id)
    os.makedirs(local_workspace_dir, exist_ok=True)
    LOGGER.info(f"Processing mask: {mask_id}")

    mask_raster_info = geoprocessing.get_raster_info(mask_raster_path)
    buffer_distance_m = max_hours * 104 * 1000  # drive 65mph for that many hours

    bbox = mask_raster_info["bounding_box"]
    buffered_bbox = box(
        bbox[0] - buffer_distance_m,
        bbox[1] - buffer_distance_m,
        bbox[2] + buffer_distance_m,
        bbox[3] + buffer_distance_m,
    )

    mask_info = geoprocessing.get_raster_info(mask_raster_path)

    # 1. Clip friction raster (native CRS)
    # friction_bbox = transform_bbox(
    #     buffered_bbox,
    #     mask_info["projection_wkt"],
    #     friction_info["projection_wkt"],
    # )
    # eckert to lat/lng always breaks, just use this
    friction_reprojected_raster_path = os.path.join(
        local_workspace_dir, "friction_reprojected.tif"
    )
    geoprocessing.warp_raster(
        FRICTION_SURFACE_RASTER_PATH,
        mask_info["pixel_size"],
        friction_reprojected_raster_path,
        "nearest",
        target_bb=buffered_bbox.bounds,
        target_projection_wkt=mask_info["projection_wkt"],
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE_OPTIONS,
    )

    # 2. Clip mask raster then reproject to friction CRS
    mask_reprojected_raster_path = os.path.join(
        local_workspace_dir, "mask_reprojected.tif"
    )
    geoprocessing.warp_raster(
        mask_raster_path,
        mask_info["pixel_size"],
        mask_reprojected_raster_path,
        "nearest",
        target_bb=buffered_bbox.bounds,
        target_projection_wkt=mask_info["projection_wkt"],
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE_OPTIONS,
    )

    # 3. Clip population raster then reproject to friction CRS
    pop_reprojected_raster_path = os.path.join(
        local_workspace_dir, "population_reprojected.tif"
    )
    geoprocessing.warp_raster(
        POPULATION_RASTER_PATH,
        mask_info["pixel_size"],
        pop_reprojected_raster_path,
        "nearest",
        target_bb=buffered_bbox.bounds,
        target_projection_wkt=mask_info["projection_wkt"],
        raster_driver_creation_tuple=GTIFF_CREATION_TUPLE_OPTIONS,
    )

    LOGGER.info(f"Completed processing for mask: {mask_id}")

    LOGGER.info(f"{mask_id}: Reading friction surface raster")
    with rasterio.open(friction_reprojected_raster_path) as friction_ds:
        friction_array = friction_ds.read(1)
        transform = friction_ds.transform
        cell_length_m = transform.a
        n_rows, n_cols = friction_array.shape

    with rasterio.open(mask_reprojected_raster_path) as src:
        mask_array = src.read(1)

    binary_mask = (mask_array >= 1).astype("int8")

    LOGGER.info(friction_array.shape)
    LOGGER.info(binary_mask.shape)
    LOGGER.info(cell_length_m)
    LOGGER.info(n_cols)
    LOGGER.info(n_rows)
    LOGGER.info(max_time_mins)

    travel_reach_array = shortest_distances.find_mask_reach(
        friction_array,
        binary_mask,
        cell_length_m,
        n_cols,
        n_rows,
        max_time_mins,
    )

    LOGGER.info(
        f"{mask_id}: Calculating travel reach raster ({n_cols} cols x {n_rows} rows)"
    )
    target_max_reach_raster_path = os.path.join(
        local_workspace_dir, f"{mask_id}_max_reach_{max_time_mins}min.tif"
    )

    LOGGER.info(
        f"{mask_id}: Writing travel reach raster to {target_max_reach_raster_path}"
    )

    with rasterio.open(friction_reprojected_raster_path) as tpl_ds:
        reach_meta = tpl_ds.meta.copy()
        reach_meta.update({"dtype": travel_reach_array.dtype, "count": 1, "nodata": 0})

    with rasterio.open(target_max_reach_raster_path, "w", **reach_meta) as dst:
        dst.write(travel_reach_array, 1)

    with rasterio.open(pop_reprojected_raster_path, "r") as pop_raster:
        pop_array = pop_raster.read(1)

    in_range_pop_count = np.sum(pop_array[(travel_reach_array > 0) & (pop_array > 0)])

    LOGGER.info(
        f"{mask_id}: Population within {max_hours} hours travel: {in_range_pop_count}"
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
    for mask_id, mask_raster_path in MASK_RASTER_PATH_DICT.items():
        for max_hours in TRAVEL_TIME_HOURS:
            in_range_pop_count_task = task_graph.add_task(
                func=process_raster_mask,
                args=(mask_id, mask_raster_path, max_hours, WORKSPACE_DIR),
                store_result=True,
                transient_run=True,
                task_name=f"process {mask_id} at {max_hours} hours",
            )
            task_list.append((mask_id, max_hours, in_range_pop_count_task))
    for mask_id, max_hours, in_range_pop_count_task in task_list:
        if mask_id not in results_dict:
            results_dict[mask_id] = {}
        results_dict[mask_id][max_hours] = in_range_pop_count_task.get()

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
