import numpy as np
import probeinterface.io
import spikeinterface.core as si
import spikeinterface.extractors as se


class LussacData:
	"""
	The main data object for Lussac.

	Attributes:
		recording 	A SpikeInterface recording object.
		sortings	A list of SpikeInterface sorting objects.
	"""

	__slots__ = "recording", "sortings", "params"
	recording: si.BaseRecording
	sortings: list[si.BaseSorting]
	params: dict

	def __init__(self, params: dict) -> None:
		"""
		Creates a new LussacData instance.
		Loads all the necessary information from spike sorters output.

		@param params: dict
			The params.json file containing everything we need to know.
		"""

		self._format_params(params)
		self.params = params

		self.recording = si.BinaryRecordingExtractor(params['recording']['file'], sampling_frequency=params['recording']['sampling_rate'],
													 num_chan=params['recording']['n_channels'], dtype=params['recording']['dtype'],
													 gain_to_uV=params['recording']['gain_uV'], offset_to_uV=params['recording']['offset_uV'])
		self._setup_probe(params['recording']['probe_file'])

		self.sortings = []
		for path in params['phy_folders']:
			self.sortings.append(se.PhySortingExtractor(path))

	@property
	def sampling_f(self) -> float:
		"""
		Returns the sampling frequency of the recording (in Hz).

		@return sampling_frequency: float
		"""

		return self.recording.get_sampling_frequency()

	@staticmethod
	def _format_params(params: dict) -> None:
		"""
		Formats the parameters the correct way.
		For example, is the value is a string containing the path to a file,
		it will read the file and replace the value with its content.

		@param params: dict
			A dictionary containing Lussac's parameters.
		"""

		if isinstance(params['recording']['gain_uV'], str):
			params['recording']['gain_uV'] = np.load(params['recording']['gain_uV'])
		if isinstance(params['recording']['offset_uV'], str):
			params['recording']['offset_uV'] = np.load(params['recording']['offset_uV'])

	def _setup_probe(self, filename: str) -> None:
		"""
		Loads the probe geometry into the 'recording' attribute.

		@param filename: str
			Path to the JSON file containing the probe geometry (ProbeInterface format).
		"""

		probe_group = probeinterface.io.read_probeinterface(filename)
		self.recording = self.recording.set_probegroup(probe_group)
