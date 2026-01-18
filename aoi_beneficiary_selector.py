"""
This pipeline computes downstream beneficiary counts for each subwatershed by
traversing the watershed network in the downstream direction and aggregating
population totals from all connected downstream units. It produces
per-subwatershed downstream population summaries that can be used to rank or
prioritize upstream areas based on the number of people affected downstream.
"""

from collections import defaultdict, deque
from pathlib import Path
import logging

from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, intersection
from shapely.ops import transform as shapely_transform
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

        subwatersheds_intersecting_pp = subwatershed_vector.iloc[
            subwatershed_candidate_row_idxs
        ].copy()

        if subwatersheds_intersecting_pp.empty:
            continue

        logging.info("gather population counts into intersecting subwatersheds")
        with rasterio.open(POPULATION_RASTER_PATH) as population_raster_ds:
            subwatersheds_raster_crs_gdf = subwatersheds_intersecting_pp.to_crs(
                population_raster_ds.crs
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

        subwatersheds_intersecting_pp["pop_sum"] = (
            subwatershed_population_sum_vals
        )

        subwatersheds_intersecting_pp.loc[:, "pop_sum"] = (
            subwatershed_population_sum_vals
        )
        # we were getting duplicate rows, this ensures we have the same HYBAS_ID
        subwatersheds_intersecting_pp = (
            subwatersheds_intersecting_pp.drop_duplicates(
                subset=["HYBAS_ID"]
            ).copy()
        )

        logging.info("build next_upstream_ids_by_subwatershed_id index to walk")
        subwatershed_id_col = "HYBAS_ID"
        downstream_id_col = "NEXT_DOWN"

        subwatersheds_by_id_gdf = subwatersheds_intersecting_pp.set_index(
            subwatershed_id_col, drop=False
        )
        subwatershed_id_set = set(subwatersheds_by_id_gdf.index.to_numpy())
        next_upstream_ids_by_subwatershed_id = defaultdict(list)
        for upstream_subwatershed_id, downstream_subwatershed_id in zip(
            subwatersheds_intersecting_pp[subwatershed_id_col].to_numpy(),
            subwatersheds_intersecting_pp[downstream_id_col].to_numpy(),
        ):
            if downstream_subwatershed_id in subwatershed_id_set:
                next_upstream_ids_by_subwatershed_id[
                    downstream_subwatershed_id
                ].append(upstream_subwatershed_id)

        intersects_pp_by_subwatershed_id = dict(
            zip(
                subwatersheds_by_id_gdf.index.to_numpy(),
                subwatersheds_by_id_gdf.geometry.intersects(
                    priority_place_union_geom
                ),
            )
        )

        priority_place_union_wgs84_geom = (
            gpd.GeoSeries(
                [priority_place_union_geom],
                crs=subwatersheds_intersecting_pp.crs,
            )
            .to_crs("EPSG:4326")
            .iloc[0]
        )
        priority_place_centroid_wgs84_geom = (
            priority_place_union_wgs84_geom.centroid
        )
        local_laea_crs = CRS.from_proj4(
            f"+proj=laea +lat_0={priority_place_centroid_wgs84_geom.y} "
            f"+lon_0={priority_place_centroid_wgs84_geom.x} "
            f"+datum=WGS84 +units=m +no_defs"
        )
        to_local_laea = Transformer.from_crs(
            subwatersheds_intersecting_pp.crs,
            local_laea_crs,
            always_xy=True,
        ).transform

        sorted_seed_subwatershed_ids = (
            subwatersheds_intersecting_pp.sort_values(
                "pop_sum", ascending=False
            )[subwatershed_id_col].to_list()
        )

        selected_pp_area_ha = 0.0
        total_pop_selected = 0
        MAX_PP_AREA_HA = 100_000_000
        visted_subwatersheds = set()

        selected_pp_subwatershed_rows = []

        seed_ptr = 0
        while selected_pp_area_ha < MAX_PP_AREA_HA:
            # find a subwatershed we haven't visted yet, but in decreasing
            # sorted order of pop
            while (
                seed_ptr < len(sorted_seed_subwatershed_ids)
                and sorted_seed_subwatershed_ids[seed_ptr]
                in visted_subwatersheds
            ):
                seed_ptr += 1
            if seed_ptr >= len(sorted_seed_subwatershed_ids):
                break

            seed_subwatershed_id = sorted_seed_subwatershed_ids[seed_ptr]
            seed_ptr += 1

            # keep walking upstream until we find a subwatershed intersecting
            # with the priorty place, clip it, add it to the set, then break
            bfs_queue = deque([seed_subwatershed_id])
            while bfs_queue:
                current_subwatershed_id = bfs_queue.popleft()
                if current_subwatershed_id in visted_subwatersheds:
                    continue
                total_pop_selected += subwatersheds_by_id_gdf.loc[
                    current_subwatershed_id
                ]["pop_sum"]
                visted_subwatersheds.add(current_subwatershed_id)

                # if it intersects the priority place, mark it and we're
                # done with this walk
                if intersects_pp_by_subwatershed_id.get(
                    current_subwatershed_id, False
                ):
                    current_subwatershed_row = subwatersheds_by_id_gdf.loc[
                        current_subwatershed_id
                    ].copy()
                    current_subwatershed_geom = subwatersheds_by_id_gdf.loc[
                        current_subwatershed_id
                    ].geometry

                    clipped_pp_geom = current_subwatershed_geom.intersection(
                        priority_place_union_geom
                    )
                    if not clipped_pp_geom.is_empty:
                        clipped_pp_area_ha = (
                            shapely_transform(
                                to_local_laea, clipped_pp_geom
                            ).area
                            / 10000.0
                        )
                        current_subwatershed_row.geometry = clipped_pp_geom
                        current_subwatershed_row["pp_area_ha"] = float(
                            clipped_pp_area_ha
                        )
                        selected_pp_subwatershed_rows.append(
                            current_subwatershed_row
                        )
                        selected_pp_area_ha += clipped_pp_area_ha
                        # we found an intersecting subwatershed, so all done
                        break
                # add the upstream subwatersheds for the next step
                for (
                    upstream_subwatershed_id
                ) in next_upstream_ids_by_subwatershed_id.get(
                    current_subwatershed_id, []
                ):
                    if upstream_subwatershed_id not in visted_subwatersheds:
                        bfs_queue.append(upstream_subwatershed_id)

        selected_pp_subwatersheds_gdf = gpd.GeoDataFrame(
            selected_pp_subwatershed_rows,
            crs=subwatersheds_intersecting_pp.crs,
        )
        final_union_geom = selected_pp_subwatersheds_gdf.geometry.union_all()
        final_result_gdf = gpd.GeoDataFrame(
            [
                {
                    "total_downstream_pop": float(total_pop_selected),
                    "total_pp_selected_area_ha": float(selected_pp_area_ha),
                    "geometry": final_union_geom,
                }
            ],
            crs=selected_pp_subwatersheds_gdf.crs,
        )

        out_path = out_dir / f"{pp_vector_path.stem}_subwatersheds.gpkg"
        logging.info(f"saving result to {str(out_path)}")
        final_result_gdf.to_file(out_path)


if __name__ == "__main__":
    main()
