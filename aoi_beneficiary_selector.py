"""
This pipeline computes downstream beneficiary counts for each subwatershed by
traversing the watershed network in the downstream direction and aggregating
population totals from all connected downstream units. It produces
per-subwatershed downstream population summaries that can be used to rank or
prioritize upstream areas based on the number of people affected downstream.
"""

from pathlib import Path
import logging

from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, intersection
import geopandas as gpd
import rasterio

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d - %(message)s",
)

logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("pyogrio").setLevel(logging.WARNING)


POPULATION_RASTER_PATH = "data/pop_rasters/landscan-global-2023.tif"

MAX_PP_BUFFER_DISTANCE_M = 500 * 1000


def main():
    subwatershed_vector_path = Path("data/dem_precondition/merged_lev06.shp")
    pp_vector_dir = Path("data/priorityPlaces/")
    out_dir = Path("output_aoi_beneficiary_selector/")
    out_dir.mkdir(parents=True, exist_ok=True)

    subwatershed_vector = gpd.read_file(subwatershed_vector_path)
    if subwatershed_vector.crs is None:
        raise ValueError("Subwatershed layer has no CRS.")

    for pp_vector_path in pp_vector_dir.iterdir():
        logging.info(f"processing {pp_vector_path}")
        try:
            pp = gpd.read_file(pp_vector_path)
        except:
            continue
        if pp.empty:
            logging.warning(
                f"{pp_vector_path} does not intersect {subwatershed_vector_path}"
            )

        if pp.crs != subwatershed_vector.crs:
            pp = pp.to_crs(subwatershed_vector.crs)

        priority_place_union_geom = pp.geometry.union_all()
        pp_crs_obj = CRS.from_user_input(pp.crs)

        if pp_crs_obj.is_projected and pp_crs_obj.axis_info[0].unit_name in (
            "metre",
            "meter",
        ):
            priority_place_buffer_geom = priority_place_union_geom.buffer(
                MAX_PP_BUFFER_DISTANCE_M
            )
        else:
            priority_place_union_wgs84_geom = (
                gpd.GeoSeries([priority_place_union_geom], crs=pp.crs)
                .to_crs("EPSG:4326")
                .iloc[0]
            )
            priority_place_centroid_wgs84_geom = (
                priority_place_union_wgs84_geom.centroid
            )

            local_aeqd_crs = CRS.from_proj4(
                f"+proj=aeqd +lat_0={priority_place_centroid_wgs84_geom.y} "
                f"+lon_0={priority_place_centroid_wgs84_geom.x} "
                f"+datum=WGS84 +units=m +no_defs"
            )

            priority_place_union_aeqd_geom = (
                gpd.GeoSeries([priority_place_union_geom], crs=pp.crs)
                .to_crs(local_aeqd_crs)
                .iloc[0]
            )
            priority_place_buffer_aeqd_geom = (
                priority_place_union_aeqd_geom.buffer(MAX_PP_BUFFER_DISTANCE_M)
            )
            priority_place_buffer_geom = (
                gpd.GeoSeries(
                    [priority_place_buffer_aeqd_geom], crs=local_aeqd_crs
                )
                .to_crs(pp.crs)
                .iloc[0]
            )

        subwatershed_candidate_row_idxs = subwatershed_vector.sindex.query(
            priority_place_buffer_geom,
            predicate="intersects",
        )

        subwatersheds_bbox_intersecting_gdf = subwatershed_vector.iloc[
            subwatershed_candidate_row_idxs
        ].copy()

        if subwatersheds_bbox_intersecting_gdf.empty:
            continue

        with rasterio.open(POPULATION_RASTER_PATH) as population_raster_ds:
            subwatersheds_raster_crs_gdf = (
                subwatersheds_bbox_intersecting_gdf.to_crs(
                    population_raster_ds.crs
                )
            )
            raster_full_window = Window(
                0, 0, population_raster_ds.width, population_raster_ds.height
            )

            subwatershed_population_sum_vals = []
            for subwatershed_geom in subwatersheds_raster_crs_gdf.geometry:
                subwatershed_window = (
                    from_bounds(
                        *subwatershed_geom.bounds,
                        transform=population_raster_ds.transform,
                    )
                    .round_offsets()
                    .round_lengths()
                )

                subwatershed_window = intersection(
                    subwatershed_window, raster_full_window
                )

                population_window_ma = population_raster_ds.read(
                    1, window=subwatershed_window, masked=True
                )
                subwatershed_window_transform = (
                    population_raster_ds.window_transform(subwatershed_window)
                )

                subwatershed_inside_mask = geometry_mask(
                    [subwatershed_geom],
                    out_shape=population_window_ma.shape,
                    transform=subwatershed_window_transform,
                    invert=True,
                    all_touched=True,
                )

                subwatershed_population_sum = population_window_ma.filled(0)[
                    subwatershed_inside_mask
                ].sum()
                subwatershed_population_sum_vals.append(
                    float(subwatershed_population_sum)
                )
        subwatersheds_bbox_intersecting_gdf["pop_sum"] = (
            subwatershed_population_sum_vals
        )

        out_path = out_dir / f"{pp_vector_path.stem}_subwatersheds.shp"
        logging.info(f"saving result to {str(out_path)}")
        subwatersheds_bbox_intersecting_gdf.to_file(out_path)


if __name__ == "__main__":
    main()
