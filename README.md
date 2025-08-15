# Ecosystem Service Beneficiaries Analysis

This repository contains scripts to run the ecosystem service beneficiaries analysis for Roadmap 2030 indicators and other related WWF initiatives.
The goal is to calculate the number of people who may potentially benefit from WWF's projects by assessing spatial relationships between project areas and nearby populations.

## Overview

The analysis contains the following scripts

- **downstream_beneficiaries.py.py** — delineates and counts populations located downstream of areas that have improved or declined in hydrologic ecosystem services within WWF project areas
- **nature_access_beneficiaries.py** — delineates and counts populations within an hour's travel of WWF project areas, according to a friction surface, as a proxy for any cultural or material services that are experienced or gathered by people visiting the location
- **combine_beneficiaries.py** — delineates and counts the overlap between the two (since many populations will benefit from both types of services, and we don't want to double count them)



## Dependencies

The scripts run in a docker container that can be built through this command:

docker build . -t therealspring/wwf_es_beneficiaries_executor:latest && docker run --rm -it -v .:/usr/local/wwf_es_beneficiaries therealspring/wwf_es_beneficiaries_executor:latest

Once built the docker container can just be run:
docker run --rm -it -v .:/usr/local/wwf_es_beneficiaries therealspring/wwf_es_beneficiaries_executor:latest


## Getting Started

1. Clone this repository:
   git clone https://github.com/springinnovate/wwf_es_beneficiaries.git

2. Set up the data according to this architecture:
```
    wwf_es_beneficiaries/ (this repo)
        data/
            aois/
            dem_precondition/
            es_change_rasters/
            pop_rasters/
            travel_time/
```

(Data can be downloaded here: https://worldwildlifefund.sharepoint.com/sites/RM2030Pilot-GeospatialAnalysis/_layouts/15/guestaccess.aspx?share=Eqn9a_rQlMhIppbWxsSBni0BbDss69JBmVdvAFF_PdD-oA&e=Lmvcxr)

3. Run the scripts within the docker environment from the 
    python downstream_beneficiaries.py
    python nature_access_beneficiaries.py
    combine_beneficiaries.py

4. Outputs include folders of masks of the downstream and within-1-hr areas and tables summing the populations by project areas.
