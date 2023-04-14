import sys
import argparse
import json
import jsmin
from lussac.core.lussac_data import LussacData
from lussac.core.pipeline import LussacPipeline


def parse_arguments(args: list | None) -> str:
	"""
	Parses the arguments given when launching Lussac.

	@param args: list
		All the arguments to parse.
	@return params_file: str
		Path to the file containing Lussac's parameters.
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("params_file", help="Path to the json file containing Lussac's parameters.")
	args = parser.parse_args(args)

	return args.params_file


def load_json(filename: str) -> dict:
	"""
	Loads the JSON parameters file and returns its content.

	@param filename: str
		Path to the file containing Lussac's parameters.
	@return params: dict
		Lussac's parameters.
	"""

	folder = filename[:filename.rindex('/')] if '/' in filename else "./"
	with open(filename) as json_file:
		minified = jsmin.jsmin(json_file.read())  # Parses out comments.
		minified = minified.replace("$PARAMS_FOLDER", folder)
		return json.loads(minified)


def main() -> None:  # pragma: no cover
	"""
	The main function to execute Lussac.
	"""

	# STEP 0: Loading the parameters into the LussacData object.
	params_file = parse_arguments(sys.argv[1:])
	params = load_json(params_file)
	data = LussacData.create_from_params(params)

	# STEP 1: Running the spike sorting.
	pass

	# STEP 2: Running the pipeline.
	pipeline = LussacPipeline(data)
	pipeline.launch()


if __name__ == "__main__":  # pragma: no cover
	main()