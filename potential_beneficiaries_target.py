"""
This pipeline computes downstream beneficiary counts for each subwatershed by
traversing the watershed network in the downstream direction and aggregating
population totals from all connected downstream units. It produces
per-subwatershed downstream population summaries that can be used to rank or
prioritize upstream areas based on the number of people affected downstream.
"""

from collections import defaultdict, deque
from pathlib import Path
import argparse
import logging
import yaml

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


def main():
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base = args.config.parent

    aggregate_vector_path = Path(cfg["aggregate_vector_path"])
    if not aggregate_vector_path.is_absolute():
        aggregate_vector_path = (base / aggregate_vector_path).resolve()
    aggregate_vector = gpd.read_file(aggregate_vector_path)
    if aggregate_vector.crs is None:
        raise ValueError("Subwatershed layer has no CRS.")

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = (base / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for focal_vector_entry in cfg["focal_vectors"]:
        focal_id = focal_vector_entry["id"]

        focal_vector_path = Path(focal_vector_entry["path"])
        if not focal_vector_path.is_absolute():
            focal_vector_path = (base / focal_vector_path).resolve()

        is_marine = bool(focal_vector_entry["is_marine"])
        max_pp_buffer_distance_m = float(
            focal_vector_entry["buffer_distance_m"]
        )

        try:
            focal_vector = gpd.read_file(focal_vector_path)
        except:
            continue
        if focal_vector.empty:
            logging.warning(
                f"{focal_vector_path} does not intersect {aggregate_vector_path}"
            )

        if focal_vector.crs != aggregate_vector.crs:
            focal_vector = focal_vector.to_crs(aggregate_vector.crs)

        priority_place_union_geom = focal_vector.geometry.union_all()
        focal_vector_crs_obj = CRS.from_user_input(focal_vector.crs)

        # buffer out the focal vector the MAX_PP_BUFFER_DISTANCE so it picks
        # up the possible polygons to start with
        if focal_vector_crs_obj.is_projected and focal_vector_crs_obj.axis_info[
            0
        ].unit_name in (
            "metre",
            "meter",
        ):
            priority_place_buffer_geom = priority_place_union_geom.buffer(
                max_pp_buffer_distance_m
            )
        else:
            priority_place_union_wgs84_geom = (
                gpd.GeoSeries([priority_place_union_geom], crs=focal_vector.crs)
                .to_crs("EPSG:4326")
                .iloc[0]
            )
            priority_place_centroid_wgs84_geom = (
                priority_place_union_wgs84_geom.centroid
            )

            local_aeqd_crs = CRS.from_proj4(
                f"+proj=aeqd +lat_0={priority_place_centroid_wgs84_geom.y} "
                f"+lon_0={priority_place_centroid_wgs84_geom.x} +datum=WGS84 +units=m +no_defs +over"
            )

            priority_place_union_aeqd_geom = (
                gpd.GeoSeries([priority_place_union_geom], crs=focal_vector.crs)
                .to_crs(local_aeqd_crs)
                .iloc[0]
            )
            priority_place_buffer_geom = (
                gpd.GeoSeries(
                    [priority_place_union_aeqd_geom], crs=local_aeqd_crs
                )
                .buffer(max_pp_buffer_distance_m)
                .to_crs(focal_vector.crs)
                .buffer(0)
            ).iloc[0]

        if is_marine:
            priority_place_union_geom = priority_place_buffer_geom

            subwatershed_candidate_row_idxs = aggregate_vector.sindex.query(
                priority_place_union_geom,
                predicate="intersects",
            )

            subwatershed_candidates = aggregate_vector.iloc[
                subwatershed_candidate_row_idxs
            ]

            diced_parts = subwatershed_candidates.geometry.intersection(
                priority_place_union_geom
            )
            diced_parts = diced_parts[~diced_parts.is_empty]
            # diced_gdf = subwatershed_candidates.loc[diced_parts.index].copy()
            # diced_gdf.geometry = diced_parts
            # diced_gdf.to_file(
            #     "diced_parts.gpkg", layer="diced_parts", driver="GPKG"
            # )

            priority_place_union_geom = diced_parts.unary_union
        else:
            subwatershed_candidate_row_idxs = aggregate_vector.sindex.query(
                priority_place_buffer_geom,
                predicate="intersects",
            )

        subwatersheds_intersecting_focal_vector = aggregate_vector.iloc[
            subwatershed_candidate_row_idxs
        ].copy()

        if subwatersheds_intersecting_focal_vector.empty:
            continue

        logging.info("gather population counts into intersecting subwatersheds")
        with rasterio.open(POPULATION_RASTER_PATH) as population_raster_ds:
            subwatersheds_raster_crs_gdf = (
                subwatersheds_intersecting_focal_vector.to_crs(
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

        subwatersheds_intersecting_focal_vector["pop_sum"] = (
            subwatershed_population_sum_vals
        )

        subwatersheds_intersecting_focal_vector.loc[:, "pop_sum"] = (
            subwatershed_population_sum_vals
        )
        # we were getting duplicate rows, this ensures we have the same HYBAS_ID
        subwatersheds_intersecting_pp = (
            subwatersheds_intersecting_focal_vector.drop_duplicates(
                subset=["HYBAS_ID"]
            ).copy()
        )

        logging.info("build next_upstream_ids_by_subwatershed_id index to walk")
        subwatershed_id_col = "HYBAS_ID"
        downstream_id_col = "NEXT_DOWN"

        subwatersheds_by_id_gdf = (
            subwatersheds_intersecting_focal_vector.drop_duplicates(
                subwatershed_id_col, keep="first"
            ).set_index(subwatershed_id_col, drop=False)
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
                crs=subwatersheds_intersecting_focal_vector.crs,
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
            subwatersheds_intersecting_focal_vector.crs,
            local_laea_crs,
            always_xy=True,
        ).transform

        sorted_seed_subwatershed_ids = (
            subwatersheds_intersecting_focal_vector.sort_values(
                "pop_sum", ascending=False
            )[subwatershed_id_col].to_list()
        )

        selected_pp_area_ha = 0.0
        total_pop_selected = 0
        MAX_PP_AREA_HA = 100_000_000
        visted_subwatersheds = set()

        selected_pp_subwatershed_rows = []

        seed_ptr = 0
        max_proposed_priorty_area_ha = focal_vector_entry[
            "max_proposed_priority_area_ha"
        ]
        while selected_pp_area_ha < max_proposed_priorty_area_ha:
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
            crs=subwatersheds_intersecting_focal_vector.crs,
        )
        final_union_geom = selected_pp_subwatersheds_gdf.geometry.union_all()
        final_result_gdf = gpd.GeoDataFrame(
            [
                {
                    "total_downstream_pop": int(total_pop_selected),
                    "total_pp_selected_area_ha": int(selected_pp_area_ha),
                    "geometry": final_union_geom,
                }
            ],
            crs=selected_pp_subwatersheds_gdf.crs,
        )

        out_path = out_dir / f"{focal_id}_proposed_areas.gpkg"
        logging.info(f"saving result to {str(out_path)}")
        final_result_gdf.to_file(out_path)


if __name__ == "__main__":
    main()
