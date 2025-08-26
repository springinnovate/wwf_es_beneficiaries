"""
Run with docker container:

docker run --rm -it -v .:/usr/local/roadmap2030 therealspring/roadmap2030_executor:latest
"""

import collections
import csv
import datetime
import glob
import hashlib
import itertools
import logging
import os
import sys
import tempfile
from itertools import islice

from ecoshard.geoprocessing.geoprocessing_core import (
    DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
    DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
)
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


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s"
    ),
)
LOGGER = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("ecoshard").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARN)


POP_PIXEL_SIZE = [0.008333333333333, -0.008333333333333]
km_in_deg = POP_PIXEL_SIZE[0] * 1000 / 900  # because it's 900m so converting to 1km
BUFFER_AMOUNTS_IN_PIXELS_M = [
    (int(np.round(x * km_in_deg / POP_PIXEL_SIZE[0])), x * 1000) for x in range(1, 2)
]

# This is relative because Docker will map a volume
GLOBAL_SUBWATERSHEDS_VECTOR_PATH = "./dem_precondition/merged_lev06.shp"
DEM_RASTER_PATH = "./dem_precondition/astgtm_compressed.tif"

AOI_DIRS = ["./data/WWF-Int_Pilot-sites", "./data/aoi_by_country"]

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

POPULATION_RASTERS = {
    "landscan-global-2023": "./data/pop_rasters/landscan-global-2023.tif",
}

ES_RASTERS = {
    "sed_export_tnc_ESA_2020-1992_change": "./data/ABUNCHASERVICES/sed_export_tnc_ESA_2020-1992_change_md5_0ab0cf.tif",
    "n_export_tnc_2020-1992_change": "./data/ABUNCHASERVICES/n_export_tnc_2020-1992_change_val_md5_18a2b3.tif",
}


OUTPUT_DIR = "./workspace_downstream_es_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def align_and_resize_raster_stack(
    base_raster_path_list,
    target_raster_path_list,
    resample_method_list,
    target_pixel_size,
    bounding_box_mode,
    base_vector_path_list=None,
    raster_align_index=None,
    base_projection_wkt_list=None,
    target_projection_wkt=None,
    vector_mask_options=None,
    gdal_warp_options=None,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
    osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
):
    """Generate rasters from a base such that they align geospatially.

    This function resizes base rasters that are in the same geospatial
    projection such that the result is an aligned stack of rasters that have
    the same cell size, dimensions, and bounding box. This is achieved by
    clipping or resizing the rasters to intersected, unioned, or equivocated
    bounding boxes of all the raster and vector input.

    Args:
        base_raster_path_list (sequence): a sequence of base raster paths that
            will be transformed and will be used to determine the target
            bounding box.
        target_raster_path_list (sequence): a sequence of raster paths that
            will be created to one-to-one map with ``base_raster_path_list``
            as aligned versions of those original rasters. If there are
            duplicate paths in this list, the function will raise a ValueError.
        resample_method_list (sequence): a sequence of resampling methods
            which one to one map each path in ``base_raster_path_list`` during
            resizing.  Each element must be one of
            "near|bilinear|cubic|cubicspline|lanczos|mode".
        target_pixel_size (list/tuple): the target raster's x and y pixel size
            example: (30, -30).
        bounding_box_mode (string): one of "union", "intersection", or
            a sequence of floats of the form [minx, miny, maxx, maxy] in the
            target projection coordinate system.  Depending
            on the value, output extents are defined as the union,
            intersection, or the explicit bounding box.
        base_vector_path_list (sequence): a sequence of base vector paths
            whose bounding boxes will be used to determine the final bounding
            box of the raster stack if mode is 'union' or 'intersection'.  If
            mode is 'bb=[...]' then these vectors are not used in any
            calculation.
        raster_align_index (int): indicates the index of a
            raster in ``base_raster_path_list`` that the target rasters'
            bounding boxes pixels should align with.  This feature allows
            rasters whose raster dimensions are the same, but bounding boxes
            slightly shifted less than a pixel size to align with a desired
            grid layout.  If ``None`` then the bounding box of the target
            rasters is calculated as the precise intersection, union, or
            bounding box.
        base_projection_wkt_list (sequence): if not None, this is a sequence of
            base projections of the rasters in ``base_raster_path_list``. If a
            value is ``None`` the ``base_sr`` is assumed to be whatever is
            defined in that raster. This value is useful if there are rasters
            with no projection defined, but otherwise known.
        target_projection_wkt (string): if not None, this is the desired
            projection of all target rasters in Well Known Text format. If
            None, the base SRS will be passed to the target.
        vector_mask_options (dict): optional, if not None, this is a
            dictionary of options to use an existing vector's geometry to
            mask out pixels in the target raster that do not overlap the
            vector's geometry. Keys to this dictionary are:

            * ``'mask_vector_path'`` (str): path to the mask vector file.
              This vector will be automatically projected to the target
              projection if its base coordinate system does not match the
              target.
            * ``'mask_layer_name'`` (str): the layer name to use for masking.
              If this key is not in the dictionary the default is to use
              the layer at index 0.
            * ``'mask_vector_where_filter'`` (str): an SQL WHERE string.
              This will be used to filter the geometry in the mask. Ex: ``'id
              > 10'`` would use all features whose field value of 'id' is >
              10.

        gdal_warp_options (sequence): if present, the contents of this list
            are passed to the ``warpOptions`` parameter of ``gdal.Warp``. See
            the `GDAL Warp documentation
            <https://gdal.org/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions>`_
            for valid options.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Return:
        None

    Raises:
        ValueError
            If any combination of the raw bounding boxes, raster
            bounding boxes, vector bounding boxes, and/or vector_mask
            bounding box does not overlap to produce a valid target.
        ValueError
            If any of the input or target lists are of different
            lengths.
        ValueError
            If there are duplicate paths on the target list which would
            risk corrupted output.
        ValueError
            If some combination of base, target, and embedded source
            reference systems results in an ambiguous target coordinate
            system.
        ValueError
            If ``vector_mask_options`` is not None but the
            ``mask_vector_path`` is undefined or doesn't point to a valid
            file.
        ValueError
            If ``pixel_size`` is not a 2 element sequence of numbers.

    """
    # make sure that the input lists are of the same length
    list_lengths = [
        len(base_raster_path_list),
        len(target_raster_path_list),
        len(resample_method_list),
    ]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "base_raster_path_list, target_raster_path_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths))
        )

    unique_targets = set(target_raster_path_list)
    if len(unique_targets) != len(target_raster_path_list):
        seen = set()
        duplicate_list = []
        for path in target_raster_path_list:
            if path not in seen:
                seen.add(path)
            else:
                duplicate_list.append(path)
        raise ValueError(
            "There are duplicated paths on the target list. This is an "
            "invalid state of ``target_path_list``. Duplicates: %s" % (duplicate_list)
        )

    # we can accept 'union', 'intersection', or a 4 element list/tuple
    if bounding_box_mode not in ["union", "intersection"] and (
        not isinstance(bounding_box_mode, (list, tuple)) or len(bounding_box_mode) != 4
    ):
        raise ValueError("Unknown bounding_box_mode %s" % (str(bounding_box_mode)))

    n_rasters = len(base_raster_path_list)
    if (raster_align_index is not None) and (
        (raster_align_index < 0) or (raster_align_index >= n_rasters)
    ):
        raise ValueError(
            "Alignment index is out of bounds of the datasets index: %s"
            " n_elements %s" % (raster_align_index, n_rasters)
        )

    # used to get bounding box, projection, and possible alignment info
    raster_info_list = [
        geoprocessing.get_raster_info(path) for path in base_raster_path_list
    ]

    # get the literal or intersecting/unioned bounding box
    if isinstance(bounding_box_mode, (list, tuple)):
        # if it's a sequence or tuple, it must be a manual bounding box
        LOGGER.debug("assuming manual bounding box mode of %s", bounding_box_mode)
        target_bounding_box = bounding_box_mode
    else:
        # either intersection or union, get list of bounding boxes, reproject
        # if necessary, and reduce to a single box
        if base_vector_path_list is not None:
            # vectors are only interesting for their bounding boxes, that's
            # this construction is inside an else.
            vector_info_list = [
                geoprocessing.get_vector_info(path) for path in base_vector_path_list
            ]
        else:
            vector_info_list = []

        raster_bounding_box_list = []
        for raster_index, raster_info in enumerate(raster_info_list):
            # this block calculates the base projection of ``raster_info`` if
            # ``target_projection_wkt`` is defined, thus implying a
            # reprojection will be necessary.
            if target_projection_wkt:
                if base_projection_wkt_list and base_projection_wkt_list[raster_index]:
                    # a base is defined, use that
                    base_raster_projection_wkt = base_projection_wkt_list[raster_index]
                else:
                    # otherwise use the raster's projection and there must
                    # be one since we're reprojecting
                    base_raster_projection_wkt = raster_info["projection_wkt"]
                    if not base_raster_projection_wkt:
                        raise ValueError(
                            "no projection for raster %s"
                            % base_raster_path_list[raster_index]
                        )
                # since the base spatial reference is potentially different
                # than the target, we need to transform the base bounding
                # box into target coordinates so later we can calculate
                # accurate bounding box overlaps in the target coordinate
                # system
                raster_bounding_box_list.append(
                    geoprocessing.transform_bounding_box(
                        raster_info["bounding_box"],
                        base_raster_projection_wkt,
                        target_projection_wkt,
                    )
                )
            else:
                raster_bounding_box_list.append(raster_info["bounding_box"])

        # include the vector bounding box information to make a global list
        # of target bounding boxes
        bounding_box_list = [
            (
                vector_info["bounding_box"]
                if target_projection_wkt is None
                else geoprocessing.transform_bounding_box(
                    vector_info["bounding_box"],
                    vector_info["projection_wkt"],
                    target_projection_wkt,
                )
            )
            for vector_info in vector_info_list
        ] + raster_bounding_box_list

        target_bounding_box = geoprocessing.merge_bounding_box_list(
            bounding_box_list, bounding_box_mode
        )

    if vector_mask_options:
        # ensure the mask exists and intersects with the target bounding box
        if "mask_vector_path" not in vector_mask_options:
            raise ValueError(
                "vector_mask_options passed, but no value for "
                '"mask_vector_path": %s',
                vector_mask_options,
            )

        mask_vector_info = geoprocessing.get_vector_info(
            vector_mask_options["mask_vector_path"]
        )

        if "mask_vector_where_filter" in vector_mask_options:
            # the bounding box only exists for the filtered features
            mask_vector = gdal.OpenEx(
                vector_mask_options["mask_vector_path"], gdal.OF_VECTOR
            )
            mask_layer = mask_vector.GetLayer()
            mask_layer.SetAttributeFilter(
                vector_mask_options["mask_vector_where_filter"]
            )
            mask_bounding_box = geoprocessing.merge_bounding_box_list(
                [
                    [feature.GetGeometryRef().GetEnvelope()[i] for i in [0, 2, 1, 3]]
                    for feature in mask_layer
                ],
                "union",
            )
            mask_layer = None
            mask_vector = None
        else:
            # if no where filter then use the raw vector bounding box
            mask_bounding_box = mask_vector_info["bounding_box"]

        mask_vector_projection_wkt = mask_vector_info["projection_wkt"]
        if mask_vector_projection_wkt is not None and target_projection_wkt is not None:
            mask_vector_bb = geoprocessing.transform_bounding_box(
                mask_bounding_box,
                mask_vector_info["projection_wkt"],
                target_projection_wkt,
            )
        else:
            mask_vector_bb = mask_vector_info["bounding_box"]
        # Calling `merge_bounding_box_list` will raise an ValueError if the
        # bounding box of the mask and the target do not intersect. The
        # result is otherwise not used.
        _ = geoprocessing.merge_bounding_box_list(
            [target_bounding_box, mask_vector_bb], "intersection"
        )

    if raster_align_index is not None and raster_align_index >= 0:
        # bounding box needs alignment
        align_bounding_box = raster_info_list[raster_align_index]["bounding_box"]
        align_pixel_size = raster_info_list[raster_align_index]["pixel_size"]
        # adjust bounding box so lower left corner aligns with a pixel in
        # raster[raster_align_index]
        for index in [0, 1]:
            n_pixels = int(
                (target_bounding_box[index] - align_bounding_box[index])
                / float(align_pixel_size[index])
            )
            target_bounding_box[index] = (
                n_pixels * align_pixel_size[index] + align_bounding_box[index]
            )

    for index, (base_path, target_path, resample_method) in enumerate(
        zip(base_raster_path_list, target_raster_path_list, resample_method_list)
    ):
        geoprocessing.warp_raster(
            base_path,
            target_pixel_size,
            target_path,
            resample_method,
            **{
                "target_bb": target_bounding_box,
                "raster_driver_creation_tuple": (raster_driver_creation_tuple),
                "target_projection_wkt": target_projection_wkt,
                "base_projection_wkt": (
                    None
                    if not base_projection_wkt_list
                    else base_projection_wkt_list[index]
                ),
                "vector_mask_options": vector_mask_options,
                "gdal_warp_options": gdal_warp_options,
            },
        )
        LOGGER.info(
            "%d of %d aligned: %s",
            index + 1,
            n_rasters,
            os.path.basename(target_path),
        )

    LOGGER.info("aligned all %d rasters.", n_rasters)


def clean_it(filename, max_length=1000):
    dirname, basename = os.path.split(filename)
    name, ext = os.path.splitext(basename)

    if len(dirname) + len(basename) <= max_length:
        return filename

    hash_digest = hashlib.md5(basename.encode("utf-8")).hexdigest()
    shortened_name = name[: max_length - len(hash_digest) - len(ext) - 1]
    sanitized_basename = f"{shortened_name}_{hash_digest}{ext}"

    return os.path.join(dirname, sanitized_basename)


def compute_net_and_directional_sums(
    direction_raster_path, magnitude_raster_path, target_raster_path
):
    with rasterio.open(direction_raster_path) as raster_direction, rasterio.open(
        magnitude_raster_path
    ) as magnitude_raster:
        dir_data = raster_direction.read(1, masked=True).filled(0).astype(float)
        mag_data = magnitude_raster.read(1, masked=True).filled(0).astype(float)

        net_effect = dir_data * mag_data
        positive_effect_sum = np.sum(mag_data[dir_data > 0])
        negative_effect_sum = np.sum(mag_data[dir_data < 0])

        profile = raster_direction.profile.copy()
        profile.update(dtype="float32", nodata=0)

        with rasterio.open(target_raster_path, "w", **profile) as dst:
            dst.write(net_effect.astype("float32"), 1)

        return {
            "sum": np.sum(net_effect),
            "pos_effect": positive_effect_sum,
            "neg_effect": negative_effect_sum,
        }


def calc_flow_dir(
    analysis_id,
    base_dem_raster_path,
    aoi_vector_path,
    target_clipped_dem_path,
    target_flow_dir_path,
):
    local_workspace_dir = os.path.dirname(target_flow_dir_path)
    os.makedirs(local_workspace_dir, exist_ok=True)
    dem_info = geoprocessing.get_raster_info(base_dem_raster_path)
    aoi_ds = gdal.OpenEx(aoi_vector_path, gdal.OF_VECTOR | gdal.GA_ReadOnly)
    aoi_layer = aoi_ds.GetLayer()
    minx, maxx, miny, maxy = aoi_layer.GetExtent()
    aoi_bb = [minx, miny, maxx, maxy]
    bounding_box = geoprocessing.merge_bounding_box_list(
        [dem_info["bounding_box"], aoi_bb], "intersection"
    )
    geoprocessing.warp_raster(
        base_dem_raster_path,
        POP_PIXEL_SIZE,
        target_clipped_dem_path,
        "nearest",
        target_bb=bounding_box,
        vector_mask_options={
            "mask_vector_path": aoi_vector_path,
            "all_touched": True,
            "target_mask_value": 0,
        },
    )
    r = gdal.OpenEx(target_clipped_dem_path, gdal.OF_RASTER | gdal.OF_UPDATE)
    b = r.GetRasterBand(1)
    b.SetNoDataValue(0)
    b = None
    r = None

    filled_dem_path = clean_it(
        os.path.join(local_workspace_dir, f"{analysis_id}_dem_filled.tif")
    )
    routing.fill_pits(
        (target_clipped_dem_path, 1),
        filled_dem_path,
        working_dir=local_workspace_dir,
        max_pixel_fill_count=10000,
    )

    routing.flow_dir_mfd(
        (filled_dem_path, 1),
        target_flow_dir_path,
        working_dir=local_workspace_dir,
    )


def rasterize(aoi_vector_path, dem_raster_path, aoi_raster_mask_path):
    geoprocessing.new_raster_from_base(
        dem_raster_path,
        aoi_raster_mask_path,
        datatype=gdal.GDT_Byte,
        band_nodata_list=[-1],
    )
    geoprocessing.rasterize(aoi_vector_path, aoi_raster_mask_path, burn_values=[1])


def mask_by_nonzero_and_sum(
    raster_key, base_raster_path, mask_raster_path, target_masked_path
):
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)

    base_raster_info = geoprocessing.get_raster_info(base_raster_path)
    mask_raster_info = geoprocessing.get_raster_info(mask_raster_path)
    nodata = base_raster_info["nodata"][0]

    def _mask_by_nonzero(base_array, mask_array):
        result = base_array.copy()
        result[mask_array <= 0] = nodata
        return result

    aligned_raster_path_list = [
        clean_it(
            os.path.join(
                temp_dir,
                f"%s_{raster_key}_aligned%s" % os.path.splitext(os.path.basename(path)),
            )
        )
        for path in [base_raster_path, mask_raster_path]
    ]
    align_and_resize_raster_stack(
        [base_raster_path, mask_raster_path],
        aligned_raster_path_list,
        ["near"] * 2,
        mask_raster_info["pixel_size"],
        mask_raster_info["bounding_box"],
    )

    error = False
    for path in aligned_raster_path_list:
        if not os.path.exists(path):
            LOGGER.error(
                f"{path} does not exist but the command to make it has executed"
            )
            error = True
    if error:
        raise RuntimeError(
            f"calling mask by nonzero and um like this caused a crash: "
            f"{raster_key}, {base_raster_path}, {mask_raster_path}, "
            f"{target_masked_path}"
        )

    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_raster_path_list],
        _mask_by_nonzero,
        target_masked_path,
        base_raster_info["datatype"],
        nodata,
        allow_different_blocksize=True,
        skip_sparse=True,
    )
    for raster_path in aligned_raster_path_list:
        os.remove(raster_path)

    array = gdal.OpenEx(target_masked_path).ReadAsArray()
    array = array[array != nodata]
    return np.sum(array)


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


def _chunks(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])


def subset_subwatersheds(
    aoi_vector_path, subwatershed_vector_path, subset_subwatersheds_vector_path
):
    LOGGER.info(f"processing aoi {aoi_vector_path}")
    aoi_gdf = gpd.read_file(aoi_vector_path)
    aoi_gdf["geometry"] = aoi_gdf["geometry"].buffer(0)
    aoi_union = aoi_gdf.unary_union
    aoi_bbox = box(*aoi_gdf.total_bounds)
    aoi_crs = aoi_gdf.crs

    LOGGER.info(f"Sub‑watershed CRS & ID ->NEXT mapping  {aoi_vector_path}")
    with fiona.open(subwatershed_vector_path, "r") as src:
        sub_crs = src.crs
        layer_name = src.name
        hybas_to_nextdown = {
            f["properties"]["HYBAS_ID"]: f["properties"]["NEXT_DOWN"] for f in src
        }

    LOGGER.info(f"Reproject AOI to sub‑watershed CRS {aoi_vector_path}")
    if aoi_crs != sub_crs:
        tform = pyproj.Transformer.from_crs(aoi_crs, sub_crs, always_xy=True).transform
        aoi_bbox = transform(tform, aoi_bbox)
        aoi_union = transform(tform, aoi_union)

    LOGGER.info(f"Fast spatial pre-filter {aoi_vector_path}")
    sub_bbox_gdf = gpd.read_file(subwatershed_vector_path, bbox=aoi_bbox.bounds)
    hits = sub_bbox_gdf.sindex.query(aoi_union, predicate="intersects")
    initial = sub_bbox_gdf.iloc[hits]

    LOGGER.info(f"Downstream BFS {aoi_vector_path}")
    all_ids = set(initial["HYBAS_ID"])
    queue = set(initial["NEXT_DOWN"]) - {0}
    while queue:
        all_ids.update(queue)
        queue = (
            {hybas_to_nextdown.get(hid) for hid in queue if hid in hybas_to_nextdown}
            - all_ids
            - {0}
        )

    if not all_ids:
        gpd.GeoDataFrame({"geometry": []}, crs=aoi_crs).to_file(
            subset_subwatersheds_vector_path, driver="GPKG"
        )
        LOGGER.warning("No valid geometry found; wrote empty GPKG.")
        return

    LOGGER.info(f"Fetch geometries with attribute SQL {aoi_vector_path}")
    gdf_parts = []
    for id_chunk in _chunks(all_ids, 1000):
        id_list = ",".join(map(str, id_chunk))
        sql = f'SELECT * FROM "{layer_name}" WHERE HYBAS_ID IN ({id_list})'
        gdf_parts.append(gpd.read_file(subwatershed_vector_path, sql=sql))
    sub_gdf = gpd.GeoDataFrame(pd.concat(gdf_parts, ignore_index=True), crs=sub_crs)

    LOGGER.info(f"Repair only invalid geometries {aoi_vector_path}")
    invalid = ~sub_gdf.geometry.is_valid
    if invalid.any():
        sub_gdf.loc[invalid, "geometry"] = sub_gdf.loc[invalid, "geometry"].buffer(0)

    LOGGER.info(f"Reproject to AOI CRS & write {aoi_vector_path}")
    if sub_crs != aoi_crs:
        sub_gdf = sub_gdf.to_crs(aoi_crs)
    sub_gdf.to_file(subset_subwatersheds_vector_path, driver="GPKG")
    LOGGER.info(f"all done subwatershedding {aoi_vector_path}")


# def subset_subwatersheds(
#     aoi_vector_path, subwatershed_vector_path, subset_subwatersheds_vector_path
# ):
#     # Prepare AOI
#     aoi_vector = gpd.read_file(aoi_vector_path)
#     aoi_vector.geometry = aoi_vector.geometry.buffer(0)
#     aoi_crs = aoi_vector.crs
#     aoi_union = aoi_vector.geometry.union_all()
#     aoi_bbox_geom = box(*aoi_vector.total_bounds)

#     # Retrieve subwatershed CRS
#     with fiona.open(subwatershed_vector_path, "r") as subwatershed_vector:
#         subwatershed_crs = subwatershed_vector.crs

#     if aoi_crs != subwatershed_crs:
#         transformer = pyproj.Transformer.from_crs(
#             aoi_crs, subwatershed_crs, always_xy=True
#         ).transform
#         aoi_bbox_geom = transform(transformer, aoi_bbox_geom)

#     # Initial filter based on bbox for fast lookup, then slower geometry
#     # itersection
#     subwatershed_filtered = gpd.read_file(
#         subwatershed_vector_path, bbox=aoi_bbox_geom.bounds
#     )
#     initial_subwatersheds = subwatershed_filtered[
#         subwatershed_filtered.intersects(aoi_union)
#     ]

#     all_hybas_ids = set(initial_subwatersheds["HYBAS_ID"])
#     downstream_ids = set(initial_subwatersheds["NEXT_DOWN"]) - {0}

#     # Create lookup of ID -> NEXT_DOWN
#     with fiona.open(subwatershed_vector_path, "r") as src:
#         hybas_to_nextdown = {
#             f["properties"]["HYBAS_ID"]: f["properties"]["NEXT_DOWN"] for f in src
#         }

#     # breadth first graph walk to pick up all the ids
#     while downstream_ids:
#         all_hybas_ids.update(downstream_ids)
#         downstream_ids = (
#             {
#                 hybas_to_nextdown[hybas_id]
#                 for hybas_id in downstream_ids
#                 if hybas_id in hybas_to_nextdown
#             }
#             - all_hybas_ids
#             - {0}
#         )

#     # Load all relevant subwatersheds in one go
#     with fiona.open(subwatershed_vector_path, "r") as subwatershed_vector:
#         fetched_subwatersheds = []
#         for f in subwatershed_vector:
#             if f["properties"]["HYBAS_ID"] in all_hybas_ids:
#                 geom = f.get("geometry")
#                 # there's a lot of double checking I do here because some of
#                 # the geometries we get seem to be None or have some other
#                 # invalid values?
#                 if geom is None:
#                     continue  # Skip features with no geometry at all
#                 shapely_geom = shape(geom)
#                 if shapely_geom.is_empty:
#                     continue  # Skip empty geometries
#                 fetched_subwatersheds.append(f)

#     # Explicit guard to handle empty geometries clearly
#     if not fetched_subwatersheds:
#         LOGGER.warning(
#             f"No valid geometry found in fetched subwatersheds for AOI {aoi_vector_path}. "
#             f"Creating an empty GeoDataFrame."
#         )

#         # Create a valid but empty GeoDataFrame with the correct CRS and columns
#         all_subwatersheds_gdf = gpd.GeoDataFrame({"geometry": []}, crs=subwatershed_crs)

#         if subwatershed_crs != aoi_crs:
#             all_subwatersheds_gdf = all_subwatersheds_gdf.to_crs(aoi_crs)

#         all_subwatersheds_gdf.to_file(subset_subwatersheds_vector_path, driver="GPKG")
#         return  # gracefully exit after writing empty GPKG

#     # Continue with existing logic if valid geometries exist
#     all_subwatersheds_gdf = gpd.GeoDataFrame.from_features(
#         fetched_subwatersheds, crs=subwatershed_crs
#     )

#     all_subwatersheds_gdf.geometry = all_subwatersheds_gdf.geometry.buffer(0)

#     if subwatershed_crs != aoi_crs:
#         all_subwatersheds_gdf = all_subwatersheds_gdf.to_crs(aoi_crs)

#     all_subwatersheds_gdf.to_file(subset_subwatersheds_vector_path, driver="GPKG")


MASKED_SET = set()


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(
        OUTPUT_DIR, max(1, os.cpu_count() // 2), reporting_interval=10.0
    )
    kernel_task_map = {}
    result = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict))
    )
    file = open("log.txt", "w")
    clipped_dem_work_list = []

    # prep for each analysis AOI, that means figuring out the downstream
    # subwatersheds and clipping/routing the DEM to those
    for analysis_id, aoi_vector_path in ANALYSIS_AOIS.items():
        subset_subwatersheds_vector_path = None

        local_workspace_dir = clean_it(os.path.join(OUTPUT_DIR, analysis_id))
        os.makedirs(local_workspace_dir, exist_ok=True)

        aoi_raster_mask_path = clean_it(
            os.path.join(local_workspace_dir, f"{analysis_id}_aoi_mask.tif")
        )
        target_projection_wkt = geoprocessing.get_raster_info(DEM_RASTER_PATH)[
            "projection_wkt"
        ]
        reprojected_aoi_vector_path = (
            "%s_projected.gpkg" % os.path.splitext(aoi_raster_mask_path)[0]
        )
        reproject_task = task_graph.add_task(
            func=geoprocessing.reproject_vector,
            args=(
                aoi_vector_path,
                target_projection_wkt,
                reprojected_aoi_vector_path,
            ),
            ignore_path_list=[aoi_vector_path, reprojected_aoi_vector_path],
            target_path_list=[reprojected_aoi_vector_path],
            task_name=f"reproject {analysis_id}",
        )

        subset_subwatersheds_vector_path = clean_it(
            os.path.join(local_workspace_dir, f"subwatershed_{analysis_id}.gpkg")
        )
        subset_task = task_graph.add_task(
            func=subset_subwatersheds,
            args=(
                reprojected_aoi_vector_path,
                GLOBAL_SUBWATERSHEDS_VECTOR_PATH,
                subset_subwatersheds_vector_path,
            ),
            dependent_task_list=[reproject_task],
            target_path_list=[subset_subwatersheds_vector_path],
            task_name=f"subset subwatersheds for {analysis_id}",
        )
        flow_dir_path = clean_it(
            os.path.join(local_workspace_dir, f"{analysis_id}_mfd_flow_dir.tif")
        )
        clipped_dem_path = clean_it(
            os.path.join(local_workspace_dir, f"{analysis_id}_dem.tif")
        )
        flow_dir_task = task_graph.add_task(
            func=calc_flow_dir,
            args=(
                analysis_id,
                DEM_RASTER_PATH,
                subset_subwatersheds_vector_path,
                clipped_dem_path,
                flow_dir_path,
            ),
            dependent_task_list=[subset_task],
            target_path_list=[flow_dir_path, clipped_dem_path],
            task_name=f"calculate flow dir for {analysis_id}",
        )
        clipped_dem_work_list.append(
            (
                analysis_id,
                (
                    local_workspace_dir,
                    flow_dir_path,
                    clipped_dem_path,
                    reprojected_aoi_vector_path,
                    aoi_raster_mask_path,
                    flow_dir_task,
                ),
            )
        )

    # now schedule work for all the population/es rasters given the aoi/dem
    # clipping
    for analysis_id, (
        local_workspace_dir,
        flow_dir_path,
        clipped_dem_path,
        reprojected_aoi_vector_path,
        aoi_raster_mask_path,
        flow_dir_task,
    ) in clipped_dem_work_list:
        LOGGER.info(f"processing {analysis_id}")
        rasterize_task = task_graph.add_task(
            func=rasterize,
            args=(
                reprojected_aoi_vector_path,
                clipped_dem_path,
                aoi_raster_mask_path,
            ),
            dependent_task_list=[flow_dir_task],
            ignore_path_list=[reprojected_aoi_vector_path],
            target_path_list=[aoi_raster_mask_path],
            task_name=f"{analysis_id} raster mask",
        )

        aoi_downstream_flow_mask_path = clean_it(
            os.path.join(local_workspace_dir, f"{analysis_id}_aoi_ds_coverage.tif")
        )
        flow_accum_task = task_graph.add_task(
            func=routing.flow_accumulation_mfd,
            args=((flow_dir_path, 1), aoi_downstream_flow_mask_path),
            kwargs={"weight_raster_path_band": (aoi_raster_mask_path, 1)},
            dependent_task_list=[flow_dir_task, rasterize_task],
            target_path_list=[aoi_downstream_flow_mask_path],
            task_name=f"flow accum for {analysis_id}",
        )

        # buffer out the population and ES downstream data to a mask
        for buffer_size_in_px, buffer_size_in_m in BUFFER_AMOUNTS_IN_PIXELS_M:
            if buffer_size_in_px > 0:
                buffered_downstream_flow_mask_path = clean_it(
                    f"%s_{buffer_size_in_m}m%s"
                    % os.path.splitext(aoi_downstream_flow_mask_path)
                )
                # make a kernel raster that is a circle kernel that's all 1s
                # within buffer_size_in_px from the center
                # dimensions should be
                #    buffer_size_in_px*2+1 X buffer_size_in_px*2+1
                kernel_path = clean_it(
                    os.path.join(local_workspace_dir, f"kernel_{buffer_size_in_m}.tif")
                )
                convolve_dependent_task_list = [flow_accum_task]
                # this `if` avoids double scheduling of the same task
                if kernel_path not in kernel_task_map:
                    kernel_task = task_graph.add_task(
                        func=create_circular_kernel,
                        args=(kernel_path, buffer_size_in_px),
                        target_path_list=[kernel_path],
                        task_name=f"kernel for {kernel_path}",
                    )
                    kernel_task_map[kernel_path] = kernel_task
                    convolve_dependent_task_list.append(kernel_task)
                else:
                    convolve_dependent_task_list.append(kernel_task_map[kernel_path])
                buffer_task = task_graph.add_task(
                    func=geoprocessing.convolve_2d,
                    args=(
                        (aoi_downstream_flow_mask_path, 1),
                        (kernel_path, 1),
                        buffered_downstream_flow_mask_path,
                    ),
                    kwargs={"n_workers": 1},
                    target_path_list=[buffered_downstream_flow_mask_path],
                    dependent_task_list=convolve_dependent_task_list,
                    task_name=f"buffer {buffered_downstream_flow_mask_path}",
                )
                mask_by_nonzero_and_sum_dependent_task_list = [buffer_task]
            else:
                buffered_downstream_flow_mask_path = aoi_downstream_flow_mask_path
                mask_by_nonzero_and_sum_dependent_task_list = [flow_accum_task]

            # then for each pop and ES raster, buffer that out and sum
            for raster_id, raster_path in {
                **POPULATION_RASTERS,
                **ES_RASTERS,
            }.items():
                print(f"processing {raster_id} for {buffer_size_in_m}m buffer")
                masked_raster_path = clean_it(
                    os.path.join(
                        local_workspace_dir,
                        f"{analysis_id}_{buffer_size_in_m}m_{raster_id}.tif",
                    )
                )
                mask_by_nonzero_task = task_graph.add_task(
                    func=mask_by_nonzero_and_sum,
                    args=(
                        f"{analysis_id}_{raster_id}_{buffer_size_in_m}",
                        raster_path,
                        buffered_downstream_flow_mask_path,
                        masked_raster_path,
                    ),
                    dependent_task_list=mask_by_nonzero_and_sum_dependent_task_list,
                    target_path_list=[masked_raster_path],
                    store_result=True,
                    task_name=f"{analysis_id}_population mask",
                )
                mask_key = (
                    f"{analysis_id}_{raster_id}_{buffer_size_in_m}",
                    raster_path,
                    buffered_downstream_flow_mask_path,
                    masked_raster_path,
                )
                if mask_key in MASKED_SET:
                    raise RuntimeError(f"{mask_key} was already created somehow")
                MASKED_SET.add(mask_key)
                file.write(
                    f"{analysis_id}_{raster_id}, {raster_path}, {buffered_downstream_flow_mask_path}, {masked_raster_path}\n"
                )

                result[analysis_id][raster_id][buffer_size_in_m] = {
                    "task": mask_by_nonzero_task,
                    "target_raster_path": masked_raster_path,
                }

    for analysis_id in result:
        for pop_raster_id, es_raster_id in itertools.product(
            POPULATION_RASTERS, ES_RASTERS
        ):
            for buffer_size_in_m in result[analysis_id][pop_raster_id]:
                pop_mask_task = result[analysis_id][pop_raster_id][buffer_size_in_m][
                    "task"
                ]
                pop_mask_raster_path = result[analysis_id][pop_raster_id][
                    buffer_size_in_m
                ]["target_raster_path"]
                es_mask_task = result[analysis_id][es_raster_id][buffer_size_in_m][
                    "task"
                ]
                es_mask_raster_path = result[analysis_id][es_raster_id][
                    buffer_size_in_m
                ]["target_raster_path"]
                dbsi_raster_path = clean_it(
                    os.path.join(
                        local_workspace_dir,
                        f"{analysis_id}_{pop_raster_id}_{es_raster_id}_"
                        f"{buffer_size_in_m}m_dbsi.tif",
                    )
                )
                dbsi_task = task_graph.add_task(
                    func=compute_net_and_directional_sums,
                    args=(
                        es_mask_raster_path,
                        pop_mask_raster_path,
                        dbsi_raster_path,
                    ),
                    target_path_list=[dbsi_raster_path],
                    dependent_task_list=[pop_mask_task, es_mask_task],
                    store_result=True,
                    task_name=(
                        f"calc dbsi for {analysis_id}-{pop_raster_id}-"
                        f"{es_raster_id}-{buffer_size_in_m}m"
                    ),
                )
                result[analysis_id][f"{pop_raster_id}-{es_raster_id}"][
                    buffer_size_in_m
                ] = {"task": dbsi_task, "target_raster_path": dbsi_raster_path}

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_filename = clean_it(
        os.path.join(OUTPUT_DIR, f"analysis_results_{timestamp}.csv")
    )

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "analysis_id",
                "pop_raster_id",
                "es_raster_id",
                "buffer_m",
                "pop_sum",
                "es_sum",
                "dpsi_sum",
                "dpsi_positive_pop",
                "dpsi_negative_pop",
                "pop_path",
                "es_path",
                "dpsi_path",
            ]
        )

        for analysis_id in result:
            for pop_raster_id, es_raster_id in itertools.product(
                POPULATION_RASTERS, ES_RASTERS
            ):
                for buffer_size_in_m in result[analysis_id][pop_raster_id]:
                    LOGGER.info(
                        f"processing {analysis_id}-{pop_raster_id}-"
                        f"{es_raster_id}-{buffer_size_in_m}m"
                    )
                    pop_sum = result[analysis_id][pop_raster_id][buffer_size_in_m][
                        "task"
                    ].get()
                    es_sum = result[analysis_id][es_raster_id][buffer_size_in_m][
                        "task"
                    ].get()
                    results_dict = result[analysis_id][
                        f"{pop_raster_id}-{es_raster_id}"
                    ][buffer_size_in_m]["task"].get()
                    (dpsi_sum, dpsi_positive_pop, dpsi_negative_pop) = [
                        results_dict[key] for key in ["sum", "pos_effect", "neg_effect"]
                    ]
                    pop_path = result[analysis_id][pop_raster_id][buffer_size_in_m][
                        "target_raster_path"
                    ]
                    es_path = result[analysis_id][es_raster_id][buffer_size_in_m][
                        "target_raster_path"
                    ]
                    dpsi_path = result[analysis_id][f"{pop_raster_id}-{es_raster_id}"][
                        buffer_size_in_m
                    ]["target_raster_path"]
                    writer.writerow(
                        [
                            analysis_id,
                            pop_raster_id,
                            es_raster_id,
                            buffer_size_in_m,
                            pop_sum,
                            es_sum,
                            dpsi_sum,
                            dpsi_positive_pop,
                            dpsi_negative_pop,
                            pop_path,
                            es_path,
                            dpsi_path,
                        ]
                    )
        for analysis_id, vector_path in BAD_AOIS.items():
            writer.writerow([analysis_id, "INVALID VECTOR", vector_path])

    task_graph.join()
    task_graph.close()
    LOGGER.info(f"all done results at {output_filename}")
    file.close()


if __name__ == "__main__":
    # mask_by_nonzero_and_sum(
    #     "gindrii_landscan-global-2023_0",
    #     "./data/pop_rasters/landscan-global-2023.tif",
    #     "./workspace_downstream_es_analysis/gindrii/gindrii_aoi_ds_coverage.tif",
    #     "./workspace_downstream_es_analysis/gindrii/gindrii_0m_landscan-global-2023.tif",
    # )
    main()
