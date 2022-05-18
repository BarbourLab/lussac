import argparse
import json
import jsmin
from core.lussac_data import LussacData
from core.pipeline import Pipeline


def parse_arguments() -> str:
	"""
	Parses the arguments given when launching Lussac.

	@return params_file: str
		Path to the file containing Lussac's parameters.
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("params_file", help="Path to the json file containing Lussac's parameters.")
	args = parser.parse_args()

	return args.params_file


def load_json(filename: str) -> dict:
	"""
	Loads the JSON parameters file and returns its content.

	@param filename: str
		Path to the file containing Lussac's parameters.
	@return params: dict
		Lussac's parameters.
	"""

	params_folder = filename[:filename.rindex('/')] if '/' in filename else "./"
	with open(filename) as json_file:
		minified = jsmin.jsmin(json_file.read())  # Parses out comments
		minified = minified.replace("$PARAMS_FOLDER", params_folder)
		return json.loads(minified)


if __name__ == "__main__":
	# STEP 0: Loading the parameters
	params_file = parse_arguments()
	params = load_json(params_file)

	# STEP 1: Running the spike sorting.
	pass

	# STEP 2: Running the pipeline.
	data = LussacData(params)
	pipeline = Pipeline(data)
	pipeline.launch()
