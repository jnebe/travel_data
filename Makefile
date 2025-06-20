.PHONE: full eval

clean:
	rm -f loc_data.csv real_output.csv triple_power_model.json triplepower_model_output.csv
	rm -rf graphs/

loc_data.csv: ./preprocess.py census_data/uk_boundaries_merged_2024.csv census_data/uk_2022.csv
	uv run ./preprocess.py census_data/uk_boundaries_merged_2024.csv census_data/uk_2022.csv loc_data.csv

real_output.csv: ./convert.py loc_data.csv celltower_data/merged_uk_data.csv
	uv run ./convert.py -k -d loc_data.csv celltower_data/merged_uk_data.csv balltree real_output.csv

power_model.json: ./train.py ./gravity_model/ars.py ./gravity_model/training.py loc_data.csv real_output.csv
	uv run ./train.py --optimize real_output.csv -i 5 loc_data.csv power_model.json POWER

power_model_output.csv: ./run.py power_model.json
	uv run ./run.py power_model.json power_model_output.csv 1000000

power-eval: ./eval.py power_model_output.csv real_output.csv
	uv run ./eval.py power_model_output.csv graphs/ model -c real_output.csv real

double_power_model.json: ./train.py ./gravity_model/ars.py ./gravity_model/training.py loc_data.csv real_output.csv
	uv run ./train.py --optimize real_output.csv -i 5 loc_data.csv double_power_model.json DOUBLEPOWER

doublepower_model_output.csv: ./run.py double_power_model.json
	uv run ./run.py double_power_model.json doublepower_model_output.csv 1000000

doublepower-eval: ./eval.py doublepower_model_output.csv real_output.csv
	uv run ./eval.py doublepower_model_output.csv graphs/ model -c real_output.csv real

triple_power_model.json: ./train.py ./gravity_model/ars.py ./gravity_model/training.py loc_data.csv real_output.csv
	uv run ./train.py --optimize real_output.csv -i 5 loc_data.csv triple_power_model.json TRIPLEPOWER

triplepower_model_output.csv: ./run.py triple_power_model.json
	uv run ./run.py triple_power_model.json triplepower_model_output.csv 1000000

triplepower-eval: ./eval.py triplepower_model_output.csv real_output.csv
	uv run ./eval.py triplepower_model_output.csv graphs/ model -c real_output.csv real

full-power: loc_data.csv real_output.csv power_model.json power_model_output.csv power-eval

full-doublepower: loc_data.csv real_output.csv double_power_model.json doublepower_model_output.csv doublepower-eval

full-triplepower: loc_data.csv real_output.csv triple_power_model.json triplepower_model_output.csv triplepower-eval
	
	
	
	
	