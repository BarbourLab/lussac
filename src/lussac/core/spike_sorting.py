from dataclasses import dataclass
import logging
from pathlib import Path
import time

import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss


@dataclass(slots=True)
class LussacSpikeSorter:
	"""
	Abstract class for all spike sorting algorithms running through Lussac.
	All children must have names to recognize them, and a launch method.

	Attributes:
		recording		A SpikeInterface recording object.
		name			The name of the analysis (not the sorting algorithm).
	"""

	recording: si.BaseRecording
	name: str

	def _preprocess(self, params: dict) -> None:
		"""
		Preprocesses the recording object based on what's in the params.

		@param params: dict
			Dictionary mapping the preprocessing functions to their arguments
		"""

		for preprocess_func, arguments in params.items():
			function = getattr(spre, preprocess_func)
			self.recording = function(self.recording, **arguments)

	def launch(self, params: dict) -> si.BaseSorting:
		"""
		Launches the spike sorting algorithm and returns the sorting object.

		@param params: dict
			The parameters for the spike sorting algorithm.
		"""

		sorter_name = params['sorter_name']
		folder = Path(params['sorter_params']['folder'])

		if (folder / "provenance.pkl").exists():
			sorting = si.load(folder / "provenance.pkl")
			logging.info(f"Loading analysis '{self.name}'\n")
			assert isinstance(sorting, si.BaseSorting)
			return sorting

		if 'preprocessing' in params:
			self._preprocess(params['preprocessing'])

		logging.info(f"Running spike-sorting for analysis '{self.name}' ")

		t_start = time.perf_counter()
		sorting = ss.run_sorter(sorter_name, self.recording, **params['sorter_params'])
		sorting = scur.remove_excess_spikes(sorting.remove_empty_units(), self.recording)
		sorting.annotate(name=self.name)
		sorting.dump_to_pickle(file_path=folder / "provenance.pkl", include_properties=True)
		t_end = time.perf_counter()

		logging.info(f"(done in {utils.format_elapsed_time(t_end - t_start)})\n")

		return sorting
