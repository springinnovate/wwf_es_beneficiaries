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
import math
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


def _area_ha(geom):
    """Returns the area of a geometry in hectares.

    Args:
        geom: Shapely geometry with an `.area` attribute in square units of the
            geometry's CRS.

    Returns:
        Area in hectares (square meters / 10,000) assuming the CRS units are meters.
    """
    return geom.area / 10000.0


def _erode_union(gdf, dist_m):
    """Unions all geometries in a GeoDataFrame and erodes the result by a distance.

    The erosion is performed by applying a negative buffer to the unioned geometry.

    Args:
        gdf: GeoDataFrame whose geometries will be unioned.
        dist_m: Erosion distance in the CRS linear units (typically meters). A
            positive value erodes by buffering with `-dist_m`.

    Returns:
        A shapely geometry representing the unioned and eroded result.
    """
    geom = gdf.geometry.union_all()
    geom = geom.buffer(-dist_m)
    return geom


def erode_to_area_ha(
    selected_pp_subwatersheds_geodataframe,
    max_proposed_priority_area_hectares,
    target_linear_crs,
    area_tolerance_hectares,
    maximum_iterations,
):
    """Erodes the selected subwatershed geometry to approach a target maximum area.

    The input GeoDataFrame is reprojected to `target_linear_crs` to ensure that the
    erosion distance is applied in linear units (e.g., meters/feet). The geometry is
    then iteratively "eroded" (buffered inward) and evaluated by area in hectares.
    A bracketing step expands the erosion distance upper bound until the eroded
    geometry becomes empty or its area is less than or equal to the target maximum.
    A binary search is then used to find an erosion distance that yields an area
    within `area_tolerance_hectares` of `max_proposed_priority_area_hectares`.

    If no geometry is found within tolerance after `maximum_iterations`, the
    function falls back to returning the original union geometry (i.e., no erosion).

    The returned GeoDataFrame contains a single feature (the first row of the input),
    with its geometry replaced by the selected eroded geometry (or the fallback
    original geometry).

    Args:
        selected_pp_subwatersheds_geodataframe: GeoDataFrame containing the selected
            subwatershed polygon(s). This function uses the geometry from the first
            row as the starting geometry and also uses the GeoDataFrame in erosion
            operations.
        max_proposed_priority_area_hectares: Target maximum area (in hectares) for
            the returned geometry. If the starting geometry area is already less
            than or equal to this value, the reprojected GeoDataFrame is returned
            unchanged.
        target_linear_crs: CRS to reproject into before applying erosion distances.
            This should be a projected CRS with linear units (not a geographic CRS).
        area_tolerance_hectares: Acceptable absolute difference (in hectares) between
            the eroded geometry area and `max_proposed_priority_area_hectares` for
            early termination of the binary search.
        maximum_iterations: Maximum number of binary-search iterations to perform
            when refining the erosion distance.

    Returns:
        A single-row GeoDataFrame in `target_linear_crs` whose geometry is the best
        candidate eroded geometry found. If erosion produces an empty geometry for
        all tested distances, returns an empty-geometry GeoDataFrame with the same
        non-geometry columns and CRS. If no candidate meets tolerance within the
        iteration limit, returns the original union geometry.

    """
    subwatersheds_in_target_crs_geodataframe = (
        selected_pp_subwatersheds_geodataframe.to_crs(target_linear_crs).copy()
    )
    subwatersheds_in_target_crs_geodataframe.geometry = (
        subwatersheds_in_target_crs_geodataframe.geometry.buffer(0)
    )

    starting_area_hectares = _area_ha(
        subwatersheds_in_target_crs_geodataframe.geometry.iloc[0]
    )

    if starting_area_hectares <= max_proposed_priority_area_hectares:
        logging.info(
            f"{starting_area_hectares} is less than the max proposed {max_proposed_priority_area_hectares}"
        )
        return subwatersheds_in_target_crs_geodataframe.copy()

    erosion_distance_lower_bound = 0.0
    erosion_distance_upper_bound = 1.0

    geometry_at_upper_bound = _erode_union(
        subwatersheds_in_target_crs_geodataframe,
        erosion_distance_upper_bound,
    )
    while (not geometry_at_upper_bound.is_empty) and (
        _area_ha(geometry_at_upper_bound) > max_proposed_priority_area_hectares
    ):
        logging.info(
            f"upper_bound: {erosion_distance_upper_bound} vs lower_bound: {erosion_distance_lower_bound}"
        )
        erosion_distance_upper_bound *= 2.0
        geometry_at_upper_bound = _erode_union(
            subwatersheds_in_target_crs_geodataframe,
            erosion_distance_upper_bound,
        )

    if geometry_at_upper_bound.is_empty:
        erosion_distance_upper_bound = erosion_distance_upper_bound / 2.0
        geometry_at_upper_bound = _erode_union(
            subwatersheds_in_target_crs_geodataframe,
            erosion_distance_upper_bound,
        )
        if geometry_at_upper_bound.is_empty:
            return gpd.GeoDataFrame(
                subwatersheds_in_target_crs_geodataframe.drop(
                    columns="geometry"
                ),
                geometry=[],
                crs=subwatersheds_in_target_crs_geodataframe.crs,
            )

    original_union_geometry = (
        subwatersheds_in_target_crs_geodataframe.geometry.unary_union
    )
    best_candidate_geometry = geometry_at_upper_bound
    best_candidate_area_error_hectares = abs(
        _area_ha(best_candidate_geometry) - max_proposed_priority_area_hectares
    )
    found_geometry_within_tolerance = False

    for iteration_index in range(maximum_iterations):
        logging.info(
            f"upper_bound: {erosion_distance_upper_bound} vs lower_bound: {erosion_distance_lower_bound}"
        )

        erosion_distance_midpoint = (
            erosion_distance_lower_bound + erosion_distance_upper_bound
        ) / 2.0
        geometry_at_midpoint = _erode_union(
            subwatersheds_in_target_crs_geodataframe,
            erosion_distance_midpoint,
        )

        if geometry_at_midpoint.is_empty:
            erosion_distance_upper_bound = erosion_distance_midpoint
            continue

        midpoint_area_hectares = _area_ha(geometry_at_midpoint)
        midpoint_area_error_hectares = abs(
            midpoint_area_hectares - max_proposed_priority_area_hectares
        )

        if midpoint_area_error_hectares < best_candidate_area_error_hectares:
            best_candidate_area_error_hectares = midpoint_area_error_hectares
            best_candidate_geometry = geometry_at_midpoint

        if midpoint_area_error_hectares <= area_tolerance_hectares:
            best_candidate_geometry = geometry_at_midpoint
            found_geometry_within_tolerance = True
            break

        if midpoint_area_hectares > max_proposed_priority_area_hectares:
            erosion_distance_lower_bound = erosion_distance_midpoint
        else:
            erosion_distance_upper_bound = erosion_distance_midpoint

    if not found_geometry_within_tolerance:
        best_candidate_geometry = original_union_geometry

    output_single_feature_geodataframe = (
        subwatersheds_in_target_crs_geodataframe.iloc[:1].copy()
    )
    output_single_feature_geodataframe.loc[
        output_single_feature_geodataframe.index[0],
        output_single_feature_geodataframe.geometry.name,
    ] = best_candidate_geometry

    return output_single_feature_geodataframe


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
        # union all those features into one big blob
        union_geom = selected_pp_subwatersheds_gdf.geometry.union_all()
        selected_pp_subwatersheds_gdf = gpd.GeoDataFrame(
            [
                {
                    "pp_area_ha": float(
                        selected_pp_subwatersheds_gdf["pp_area_ha"].sum()
                    ),
                    "pop_sum": float(
                        selected_pp_subwatersheds_gdf["pop_sum"].sum()
                    ),
                    "geometry": union_geom,
                }
            ],
            crs=selected_pp_subwatersheds_gdf.crs,
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

        max_erode_iterations = 50
        area_tolerance = 0.01 * max_proposed_priorty_area_ha
        final_result_gdf = erode_to_area_ha(
            final_result_gdf,
            max_proposed_priorty_area_ha,
            local_aeqd_crs,
            area_tolerance,
            max_erode_iterations,
        )
        final_result_gdf.loc[
            final_result_gdf.index[0], "total_pp_selected_area_ha"
        ] = int(
            final_result_gdf.to_crs(local_aeqd_crs).geometry.iloc[0].area
            / 10000.0
        )

        out_path = out_dir / f"{focal_id}_proposed_areas.gpkg"
        logging.info(f"saving result to {str(out_path)}")
        final_result_gdf.to_file(out_path)


if __name__ == "__main__":
    main()
