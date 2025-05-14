# Interfaces

We have the five following tools:

- preprocess.py - which processes boundary and population data into a locations dataset
- convert.py - which uses the location dataset and processes trip data into a compliant trip dataset
- train.py - which uses the location dataset and optionally a desired trip dataset to generate and train a GravityModel
- run.py - which uses a GravityModel to simulate trips
- eval.py - which evaluates trip datasets

We require the following interfaces:

- location dataset
- trip dataset
- gravity model

## Location dataset

The location dataset is a csv file (or other polars compatible format) with 6 columns:

- name - String
- id - String
- lat - Float64
- long - Float64
- area - Float64
- population - Int64 this might be insufficient when looking at very large combined regions (INT_MAX is ~2.4 billion)

## Trip dataset

The location dataset is a csv file (or other polars compatible format) with 13 columns.
This is due to the fact that it also serialized the locations that make up a trip.

- start_name - String
- start_id - String
- start_area - Float64
- start_population - Int64
- start_lat - Float64
- start_long - Float64
- end_name - String
- end_id - String
- end_area - Float64
- end_population - Int64
- end_lat - Float64
- end_long - Float64
- distance - Float64

## Gravity Model

Json Representation generated using [jsonpickle](https://jsonpickle.github.io/index.html).