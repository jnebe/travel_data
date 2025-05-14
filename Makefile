.PHONE: test

test:
	uv run ./preprocess.py census_data/uk_boundaries_2024.csv census_data/uk_2022.csv loc_data.csv
	uv run ./convert.py loc_data.csv celltower_data/merged_uk_data.csv balltree real_output.csv
	uv run ./train.py --optimize real_output.csv loc_data.csv grav_model.json BASIC
	uv run ./run.py grav_model.json model_output.csv 1000000
	uv run ./eval.py model_output.csv graphs/ model
	uv run ./eval.py real_output.csv graphs/ real