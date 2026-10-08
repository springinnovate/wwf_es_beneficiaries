# Ecosystem Service Beneficiaries Analysis

This repository estimates potential ecosystem service beneficiaries for project
areas, generated full-raster extents, and downstream watershed partitions. The
main workflow reads a YAML configuration, prepares geospatial inputs, evaluates
one or more beneficiary masks, writes population rasters for each mask, and
summarizes the results into a CSV.

The workflow currently supports two beneficiary pathways:

- Travel-time beneficiaries: people within a configured travel-time threshold
  from an area of interest.
- Downstream beneficiaries: people downstream of raster pixels that satisfy a
  configured condition, such as positive ecosystem service change.

The input data stack is stored in this SharePoint folder:
<https://worldwildlifefund.sharepoint.com/sites/RM2030Pilot-GeospatialAnalysis/_layouts/15/guestaccess.aspx?share=Eqn9a_rQlMhIppbWxsSBni0BbDss69JBmVdvAFF_PdD-oA&e=Lmvcxr>

For a local run, copy that data stack into the repository root as `data/`.
Large data files and local run configurations are ignored by git; the tracked
example configuration is intended as the starting point for new analyses.

## Environment

Clone the repository and run commands from the repository root:

```powershell
git clone https://github.com/springinnovate/wwf_es_beneficiaries.git
cd wwf_es_beneficiaries
```

The Dockerfile is the most reproducible execution environment for this
workflow. It installs Python 3.11, GDAL, Rasterio, Fiona, GeoPandas, the local
`shortest_distances` Cython extension, and the Spring Innovate `ecoshard`
package used for raster processing and TaskGraph orchestration.

Build the image from the repository root:

```powershell
docker build -t ds_beneficiaries:latest .
```

Run the image with the repository mounted into the container:

```powershell
docker run --rm -it -v "%CD%":/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest
```

On Linux or macOS, use:

```bash
docker run --rm -it -v "$(pwd)":/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest
```

For a non-Docker local environment, use a conda or mamba environment with GDAL,
Rasterio, Fiona, GeoPandas, NumPy, Pandas, PyProj, Shapely, Cython, and tqdm.
Then install the Python-only requirements, the Cython extension, and `ecoshard`:

```powershell
pip install -r requirements.txt
pip install .\shortest_distances
pip install git+https://github.com/springinnovate/ecoshard.git
```

## Data Layout

The workflow expects project data under `data/`, organized by workflow role.

| Directory | Contents |
| --- | --- |
| `data/aois` | Area-of-interest vectors used when running specific project areas instead of a generated full-raster extent. |
| `data/dem_precondition` | DEM raster and HydroBASINS-style subwatershed vectors used for downstream routing. |
| `data/es_change_rasters` | Condition rasters used by `conditional_raster` masks, such as ecosystem service change rasters. |
| `data/pop_rasters` | Population rasters used to count beneficiaries. |
| `data/travel_time` | Friction or travel-time inputs used by `travel_time_population` masks. |

The main input paths in the tracked example configuration are:

| Input | Path |
| --- | --- |
| Population raster | `data/pop_rasters/landscan-global-2023.tif` |
| Travel-time raster | `data/travel_time/friction_surface_2019_compressed_md5_1be7dd230178a5d395529be7a5e3fb0a.tif` |
| DEM raster | `data/dem_precondition/astgtm_compressed.tif` |
| Subwatershed vector | `data/dem_precondition/merged_lev06.shp` |
| Nitrogen export change raster | `data/es_change_rasters/n_export_tnc_2020-1992_change_val_md5_18a2b3.tif` |
| Sediment export change raster | `data/es_change_rasters/sed_export_tnc_ESA_2020-1992_change_md5_0ab0cf.tif` |

The subwatershed vector must include `HYBAS_ID`, `NEXT_DOWN`, and `NEXT_SINK`.
Those fields are used to trace downstream watersheds and partition work by
terminal drain.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `workflow_runner.py` | Main YAML-driven workflow runner. This is the primary entry point for analyses. |
| `configs/example_roadmap2030_pop_downstream_analysis.yaml` | Tracked example configuration showing the expected YAML structure. |
| `shortest_distances/` | Cython extension used for travel-time reach expansion. |
| `potential_beneficiaries_target.py` | Earlier downstream beneficiary utility for subwatershed-level targeting experiments. |
| `utils/chunk_file.py` | Utility for splitting and rejoining very large files into zipped chunks. |
| `tests/` | Regression tests for workflow helper behavior. |
| `Dockerfile` | Reproducible container environment for running the workflow. |
| `requirements.txt` | Additional pip-installed Python requirements used by the container and local installs. |

## Configuration

The workflow is configured with a YAML file. The tracked example is:

```text
configs/example_roadmap2030_pop_downstream_analysis.yaml
```

The config filename stem must match `run_name`. For example,
`configs/example_roadmap2030_pop_downstream_analysis.yaml` must contain:

```yaml
run_name: example_roadmap2030_pop_downstream_analysis
```

Top-level keys:

| Key | Purpose |
| --- | --- |
| `run_name` | Name used in the timestamped output CSV. Must match the config filename stem. |
| `work_dir` | Directory for intermediate rasters, drain partitions, TaskGraph state, and debug artifacts. |
| `output_dir` | Directory for final per-mask rasters, combined rasters, and summary CSVs. |
| `inputs` | Shared input data paths and analysis settings. |
| `sections` | Analysis masks and the final combine section. |
| `logging` | Console/file logging level and optional log file path. |

Important `inputs` keys:

| Key | Purpose |
| --- | --- |
| `population_raster_path` | Raster whose positive values are counted as beneficiaries. |
| `traveltime_raster_path` | Friction or travel-time raster used by travel-time masks. |
| `dem_raster_path` | DEM used to derive multiple-flow-direction routing. |
| `subwatershed_vector_path` | HydroBASINS-style vector used to partition downstream work. |
| `aoi_vector_pattern` | One or more glob patterns for AOI vectors. Leave empty when `analyze_full_raster_extent` is true. |
| `analyze_full_raster_extent` | Set to `true` to generate one AOI from the shared extent of the population, travel-time, DEM, and condition rasters. |
| `debug_drain_index` | Optional zero-based drain partition index. When set, only one terminal-drain partition is processed for debugging. |
| `wgs84_pixel_size` | Output/alignment pixel size in WGS84 degrees. Leave this alone unless changing the data stack deliberately. |
| `travel_time_pixel_size_m` | Meter-scale pixel size used when converting travel/downstream buffer distances to pixels. |
| `buffer_size_m` | Circular buffer radius applied to downstream coverage before population is counted. |

The `sections` block has one `masks` section and one `combine` section. A
typical mask section looks like:

```yaml
sections:
  - masks:
    - id: a_within_travel_time
      type: travel_time_population
      params:
        max_hours: 2.0
    - id: n_export_mask
      type: conditional_raster
      params:
        condition_raster_path: "./data/es_change_rasters/n_export_tnc_2020-1992_change_val_md5_18a2b3.tif"
        expression: "value > 0"
        max_downstream_distance_m: 50000
  - combine:
    - logic: OR
```

`travel_time_population` masks require `max_hours`.

`conditional_raster` masks require `condition_raster_path` and `expression`.
The expression is evaluated on each raster block with `value` as the raster
array and `np` as NumPy. For example, `value > 0` selects positive pixels.
`max_downstream_distance_m` is optional.

## Execution Workflow

Run commands from the repository root unless otherwise noted.

### 1. Prepare a Config

Copy the tracked example config to a new local config file under `configs/`.
Local configs are ignored by git so analysis-specific paths and run names do
not need to be committed.

```powershell
copy .\configs\example_roadmap2030_pop_downstream_analysis.yaml .\configs\my_run.yaml
```

Update `run_name`, `work_dir`, `output_dir`, input paths, AOI patterns, and mask
definitions. Remember that the config filename stem and `run_name` must match.

### 2. Run the Workflow

In Docker:

```text
cd /usr/local/wwf_es_beneficiaries
python workflow_runner.py configs/example_roadmap2030_pop_downstream_analysis.yaml
```

In a local environment:

```powershell
python workflow_runner.py .\configs\example_roadmap2030_pop_downstream_analysis.yaml
```

At startup, the runner validates required input paths and checks that the
subwatershed vector contains `HYBAS_ID`, `NEXT_DOWN`, and `NEXT_SINK`.

### 3. Run Over the Full Raster Extent

To analyze the shared raster extent instead of AOI files, set:

```yaml
inputs:
  analyze_full_raster_extent: true
  aoi_vector_pattern:
```

The workflow generates one AOI named `full_raster_extent` from the intersecting
WGS84 bounds of the population, travel-time, DEM, and condition rasters.

### 4. Debug One Drain Partition

For downstream analyses, use `debug_drain_index` to process a single terminal
drain partition and inspect the intermediate rasters:

```yaml
inputs:
  debug_drain_index: 0
```

The index is zero-based over the sorted drain partitions. Start with `0` and
increase it to inspect different partitions. When this option is set, the
workflow writes only the selected drain partition and keeps its intermediate
workspace under:

```text
<work_dir>/<aoi>/drain_<NEXT_SINK>/
```

## Outputs

Each run writes final rasters to `output_dir`, intermediate files to `work_dir`,
and one timestamped CSV summary to `output_dir`.

| Output | Description |
| --- | --- |
| `<aoi>_<mask_id>_pop.tif` | Population raster for one mask and AOI. Pixels outside the mask are zero. |
| `<aoi>_drain_<NEXT_SINK>_<mask_id>_pop.tif` | Per-drain population raster for conditional masks when an AOI has multiple terminal-drain partitions. |
| `<aoi>_total_pop.tif` | Combined population raster across all configured masks for the AOI. |
| `<run_name>_<YYYY_MM_DD_HH_MM_SS>.csv` | Summary table with one row per AOI and one column per mask plus `combined pop`. |
| `<work_dir>/<aoi>/drain_partitions/*.gpkg` | Generated downstream drain partitions used for conditional masks. |
| `<work_dir>/<aoi>/<mask_id>/` | Travel-time intermediates for travel-time masks. |
| `<work_dir>/<aoi>/drain_<NEXT_SINK>/` | DEM, flow-direction, condition, downstream coverage, and masked-population intermediates for conditional masks. |

The final combined raster uses the maximum population value per pixel across
mask outputs. This avoids double-counting the same population pixel when two
masks overlap.

## Runtime Notes

Full-raster and global downstream runs can be large. The workflow partitions
conditional downstream work by terminal drain (`NEXT_SINK`) so independent
drain systems can run in parallel through TaskGraph. The worker count is
selected from the number of AOI or drain work units and the available physical
CPU count.

Most rasters written by the workflow use tiled, compressed GeoTIFF creation
options with 256 x 256 blocks. This keeps raster calculator and combine steps
aligned with block-window processing instead of forcing scanline reads.

The workflow logs progress with tqdm-compatible console output and periodic
heartbeat messages for long blocking steps such as travel-time expansion and
large population combines. If `logging.to_file` is set, the same log messages
are also written to that file.

## Methodology

The workflow estimates potential beneficiaries by building one population mask
per configured pathway, summing each masked population raster, and then
combining all pathway rasters without double-counting overlapping population
pixels.

### Area Setup

Each AOI is read from `aoi_vector_pattern`, or generated from the full shared
raster extent when `analyze_full_raster_extent` is true. The workflow chooses a
working projected CRS from the AOI bounds. Small areas use the appropriate UTM
zone, regional areas use an azimuthal equidistant CRS centered on the AOI, and
very large areas use a global projection strategy.

For conditional downstream masks, the AOI is intersected with the subwatershed
dataset. The workflow follows `NEXT_DOWN` links to collect all downstream
watersheds, groups those watersheds by terminal drain using `NEXT_SINK`, and
writes one drain partition GeoPackage per terminal drain. Each partition is
processed independently.

### Conditional Downstream Beneficiaries

For each `conditional_raster` mask, the workflow aligns and clips the DEM and
population raster to the drain partition. The clipped DEM is pit-filled and
converted to a multiple-flow-direction flow-direction raster with
`ecoshard.geoprocessing.routing`.

The condition raster is warped to the flow-direction grid and masked to the
drain partition geometry. The configured expression is evaluated to create a
binary source mask. That source mask is passed as the weight raster to
multiple-flow-direction flow accumulation, producing a downstream coverage
raster.

The downstream coverage raster is convolved with a circular kernel based on
`buffer_size_m`. If `max_downstream_distance_m` is provided, a GDAL proximity
distance transform is computed from the condition source pixels and used to
trim the buffered downstream coverage to pixels within the configured distance.
The distance transform is in pixels and is converted to meters with
`travel_time_pixel_size_m`.

Finally, positive population pixels are retained wherever the downstream
coverage mask is positive. All other pixels are set to zero, and the resulting
population raster is summed.

### Travel-Time Beneficiaries

For each `travel_time_population` mask, the population and travel-time rasters
are clipped around the AOI with a conservative buffer derived from `max_hours`.
The AOI is rasterized to the clipped grid, and the Cython
`shortest_distances.find_mask_reach` function expands outward from AOI pixels
through the friction surface until `max_hours` is reached.

The resulting travel-reach mask is applied to the clipped population raster.
Positive population pixels inside the reach mask are retained, all other pixels
are set to zero, and the resulting population raster is summed.

### Combining Beneficiaries

After each mask produces a population raster, the workflow streams all mask
rasters into a common WGS84 output grid and writes the per-pixel maximum value.
This represents a logical OR over beneficiary masks while avoiding double
counting where two masks identify the same people.

The output CSV reports the summed population for each individual mask and the
summed population in the combined raster.
