.PHONE: full eval

TRAIN_DEPS = ./train.py ./gravity_model/random_search/*.py ./gravity_model/training.py loc_data.csv real_output.csv
RUN_DEPS = ./run.py
EVAL_DEPS = ./eval.py real_output.csv

CMD_PREFIX := $(shell if command -v uv >/dev/null 2>&1; then echo "uv run"; else echo "python"; fi)

clean:
	rm -f loc_data.csv *_output.csv *_model.json
	rm -rf graphs/

loc_data.csv: ./preprocess.py census_data/uk_boundaries_merged_2024.csv census_data/uk_2022.csv
	$(CMD_PREFIX) ./preprocess.py census_data/uk_boundaries_merged_2024.csv census_data/uk_2022.csv loc_data.csv

real_output.csv: ./convert.py loc_data.csv celltower_data/merged_uk_data.csv
	$(CMD_PREFIX) ./convert.py -k -d loc_data.csv celltower_data/merged_uk_data.csv balltree real_output.csv

ITERATIONS ?= 5

# Model training rule (e.g. power_model.json, doublepower_model.json, etc.)
%_model.json: $(TRAIN_DEPS) ./gravity_model/models/%.py 
	$(CMD_PREFIX) ./train.py --optimize real_output.csv -i $(ITERATIONS) loc_data.csv $@ $(shell echo $* | tr a-z A-Z)

# Model run rule (e.g. power_model_output.csv)
%_model_output.csv: $(RUN_DEPS) %_model.json
	$(CMD_PREFIX) ./run.py $*_model.json $@ 1000000

# Evaluation rule (e.g. power-eval)
%-eval: $(EVAL_DEPS) %_model_output.csv
	$(CMD_PREFIX) ./eval.py -c real_output.csv real $*_model_output.csv graphs/$*/ model 

# Full workflow for a type (e.g. full-power)
full-%: loc_data.csv real_output.csv %_model.json %_model_output.csv %-eval
	@echo "Completed full workflow for $*"
