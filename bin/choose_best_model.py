#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS - STEP  - Choose Best Model for Retraining

Binary Classification of Images with Different Architectures

ComparesOptuna Studies from HPO step and finds the best HP for the classification problem.


Usage:
-sf hpo_study_chckpnt_model.pkl [hpo_study_chckpnt_model.pkl ...], --study_files hpo_study_chckpnt_model.pkl [hpo_study_chckpnt_model.pkl ...]

Example:
python choose_best_model.py -sf hpo_study_checkpoint_vgg16.pkl hpo_study_checkpoint_densenet121.pkl hpo_study_checkpoint_basicnet.pkl

"""

import argparse
import tarfile
import joblib


OUTPUT_FILE = "best_model.txt"


### ------------------------- PARSER --------------------------------

parser = argparse.ArgumentParser(description='Finds best performing model in completed HPO studies')
parser.add_argument('-sf','--study_files',  metavar='hpo_study_chckpnt_model.pkl', type=str, nargs='+', required=True, help ="Completed Optuna studies")



def get_best_hp_study(hpo_file_name):
	
	best_trial_info = {}
	study = joblib.load(hpo_file_name)
	best_trial = study.best_trial
	best_trial_info["model_name"] = study.study_name
	best_trial_info["trial_id"] = best_trial.number
	best_trial_info["value"] = best_trial.value
	best_trial_info["parameters"] = best_trial.params


	return best_trial_info


def get_best_hp_overall(best_models_hpo):
	index = 0
	best_value = 0
	i = 0
	for model_hpo in best_models_hpo:
		if model_hpo["value"] > best_value:
			index = i
			best_value = model_hpo["value"]
		i+=1

	return best_models_hpo[index]



### -------------------------MAIN--------------------------------
def main():
	args     = parser.parse_args()
	study_files = args.study_files
	best_models_hp = []

	for study_file in study_files:
		best_models_hp.append(get_best_hp_study(study_file))
	
	best_params = get_best_hp_overall(best_models_hp)

	f = open(OUTPUT_FILE, "w+")
	f.write(str(best_params))
	f.close()


	return 0

if __name__ == '__main__':
    main()
