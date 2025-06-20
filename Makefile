.PHONE: full eval

clean:
	rm -f loc_data.csv real_output.csv triple_power_model.json triplepower_model_output.csv
	rm -rf graphs/

loc_data.csv: ./preprocess.py census_data/uk_boundaries_merged_2024.csv census_data/uk_2022.csv
	uv run ./preprocess.py census_data/uk_boundaries_merged_2024.csv census_data/uk_2022.csv loc_data.csv

real_output.csv: ./convert.py loc_data.csv celltower_data/merged_uk_data.csv
	uv run ./convert.py -k -d loc_data.csv celltower_data/merged_uk_data.csv balltree real_output.csv

triple_power_model.json: ./train.py loc_data.csv real_output.csv
	uv run ./train.py --optimize real_output.csv -i 5 loc_data.csv triple_power_model.json TRIPLEPOWER

triplepower_model_output.csv: ./run.py triple_power_model.json
	uv run ./run.py triple_power_model.json triplepower_model_output.csv 1000000

triplepower_eval: ./eval.py triplepower_model_output.csv real_output.csv
	uv run ./eval.py triplepower_model_output.csv graphs/ model -c real_output.csv real

full: loc_data.csv real_output.csv triple_power_model.json triplepower_model_output.csv eval
	
	
	
	
	