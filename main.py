import shutil
import argparse
import json
from jsmin import jsmin

import phy_data
import postprocessing.pipeline
import spike_sorting.kilosort2
import spike_sorting.kilosort25
import spike_sorting.kilosort3
import spike_sorting.mountainsort3
import spike_sorting.mountainsort4
import spike_sorting.spykingcircus



if __name__ == "__main__":
	############################################
	################## STEP 0 ##################
	############################################
	# 
	# Loading the parameters.
	# 

	parser = argparse.ArgumentParser()
	parser.add_argument("params_file", help="path to the json file containing Lussac's parameters")
	args = parser.parse_args()

	params_folder = args.params_file[:args.params_file.rindex('/')] if '/' in args.params_file else "./"
	with open(args.params_file) as json_file:
		minified = jsmin(json_file.read()) # Parses out comments.
		minified = minified.replace("$PARAMS_FOLDER", params_folder)
		params = json.loads(minified)



	############################################
	################## STEP 1 ##################
	############################################
	# 
	# Running spike sorting.
	# 

	for name, params0 in params['spike_sorting'].items():
		output_folder = params0['output_folder']

		if name == "Kilosort2":
			spike_sorter = spike_sorting.kilosort2.Kilosort2(params['recording'], output_folder)
		elif name == "Kilosort2.5":
			spike_sorter = spike_sorting.kilosort25.Kilosort25(params['recording'], output_folder)
		elif name == "Kilosort3":
			spike_sorter = spike_sorting.kilosort3.Kilosort3(params['recording'], output_folder)
		elif name == "MountainSort3":
			spike_sorter = spike_sorting.mountainsort3.MountainSort3(params['recording'], output_folder)
		elif name == "MountainSort4":
			spike_sorter = spike_sorting.mountainsort4.MountainSort4(params['recording'], output_folder)
		elif name == "SpykingCircus":
			spike_sorter = spike_sorting.spykingcircus.SpykingCircus(params['recording'], output_folder)
		else:
			assert False, "Could not find spike sorter \"{0}\"".format(name)

		for spk_params in params0['runs']:
			spike_sorter.launch(spk_params['name'], spk_params['params'])
			params['phy_folders'].append("{0}/{1}".format(output_folder, spk_params['name']))

		del spike_sorter



	############################################
	################## STEP 2 ##################
	############################################
	# 
	# Running post-processing.
	# 

	data = phy_data.PhyData(params)
	
	postprocessing.pipeline.launch(data, params['post_processing']['pipeline'], params['recording']['prb'])

	shutil.rmtree(params['post_processing']['tmp_folder'])

