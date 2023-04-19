from dataclasses import dataclass
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss


@dataclass(slots=True)
class LussacSpikeSorter:
	"""
	Abstract class for all spike sorting algorithms running through Lussac.
	All children must have names to recognize them, and a launch method.

	Attributes:
		recording		A SpikeInterface recording object.
	"""

	recording: si.BaseRecording

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

		sorter_name = params['sorter']

		if 'preprocessing' in params:
			self._preprocess(params['preprocessing'])

		return ss.run_sorter(sorter_name, self.recording, **params['sorter_params'])
