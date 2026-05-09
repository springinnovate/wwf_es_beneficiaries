# Ecosystem Service Beneficiaries Analysis

This repository contains code to run ecosystem service beneficiaries analyses for a variety of projects and use cases. The goal is to calculate the number of people who may potentially benefit from a given set of project areas by assessing spatial relationships between those areas and nearby populations—for example, people located downstream or people located within a specified travel-time distance.

## Overview

The analysis is orchestrated by a workflow runner that takes a YAML configuration file describing the analysis to execute. You'll need to first clone the repo, get the data, build the docker container, design your own workflow configuration `yml` file, then you can run any scripts.

## Getting Started

1. Clone this repository:

       git clone https://github.com/springinnovate/wwf_es_beneficiaries.git
       cd wwf_es_beneficiaries

2. Set up the data according to this structure:

       wwf_es_beneficiaries/ (this repo)
           data/
               aois/
               dem_precondition/
               es_change_rasters/
               pop_rasters/
               travel_time/

   Data can be downloaded here:
   https://worldwildlifefund.sharepoint.com/sites/RM2030Pilot-GeospatialAnalysis/_layouts/15/guestaccess.aspx?share=Eqn9a_rQlMhIppbWxsSBni0BbDss69JBmVdvAFF_PdD-oA&e=Lmvcxr

3. Build and run the Docker image:

       docker build . -t therealspring/wwf_es_beneficiaries_executor:latest
       docker run --rm -it -v .:/usr/local/wwf_es_beneficiaries therealspring/wwf_es_beneficiaries_executor:latest

4. Inside the container, run the desired workflow:

       cd /usr/local/wwf_es_beneficiaries
       python workflow_runner.py example_roadmap2030_pop_downstream_analysis.yaml

Use different YAML configuration files to run other analyses as needed.

To run over the full overlapping raster extent instead of AOI vector files, set
`inputs.analyze_full_raster_extent: true` and leave `aoi_vector_pattern` empty.
The workflow will generate one AOI named `full_raster_extent` from the shared
bounds of the population, travel-time, DEM, and condition rasters.
