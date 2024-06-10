import argparse
import logging
import sys

from lussac.core import LussacData, LussacParams, LussacPipeline, LussacSpikeSorter


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


def main() -> None:  # pragma: no cover
	"""
	The main function to execute Lussac.
	"""

	# STEP 0: Loading the parameters into the LussacData object.
	params_file = parse_arguments(sys.argv[1:])
	params = LussacParams.load_from_json_file(params_file)
	data = LussacData.create_from_params(params)

	# STEP 1: Running the spike sorting.
	if 'spike_sorting' in params:
		logging.info("Running spike-sorting algorithms:\n")
		for name, params0 in params['spike_sorting'].items():
			spike_sorter = LussacSpikeSorter(data.recording, name)
			data.sortings[name] = spike_sorter.launch(params0)

	# STEP 2: Running the pipeline.
	pipeline = LussacPipeline(data)
	pipeline.launch()


if __name__ == "__main__":  # pragma: no cover
	main()
