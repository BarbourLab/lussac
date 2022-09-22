import os
import pathlib
import copy
import tempfile
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import probeinterface.io
import spikeinterface.core as si
import spikeinterface.extractors as se


class LussacData:
	"""
	The main data object for Lussac.

	Attributes:
		recording 		A SpikeInterface recording object.
		sortings		A list of SpikeInterface sorting objects.
		params			A dictionary containing all of Lussac's parameters.
		_tmp_directory	The TemporaryDirectory object for Lussac.
	"""

	__slots__ = "recording", "sortings", "params", "_tmp_directory"
	recording: si.BaseRecording
	sortings: dict[str, si.BaseSorting]
	params: dict[str, dict]
	_tmp_directory: tempfile.TemporaryDirectory

	def __init__(self, recording: si.BaseRecording, sortings: dict[str, si.BaseSorting], params: dict[str, dict]) -> None:
		"""
		Creates a new LussacData instance.
		Loads all the necessary information from spike sorters output.

		@param recording: BaseRecording
			The recording object.
		@param sortings: list[BaseSorting]
			A list containing all the sorting objects (i.e. all the analyses).
		@param params: dict
			The params.json file containing everything we need to know.
		"""

		recording.annotate(is_filtered=True)  # Otherwise SpikeInterface is annoying.

		self.recording = recording
		self.sortings = sortings
		params['lussac']['pipeline'] = self._format_params(params['lussac']['pipeline'])
		self.params = params
		self._tmp_directory = self._setup_tmp_directory(params['lussac']['tmp_folder'])

	@property
	def tmp_folder(self) -> str:
		"""
		Returns the path to the temporary folder.
		This folder is deleted when Lussac exists (normally or because of a crash).

		@return tmp_folder: str
			Path to the folder that store temporary files.
		"""

		return self._tmp_directory.name

	@property
	def logs_folder(self) -> str:
		"""
		Returns the path to the logs folder.

		@return: logs_folder: str
			Path to the folder that store the logs.
		"""

		if not os.path.exists(logs_folder := self.params['lussac']['logs_folder']):
			os.makedirs(logs_folder)

		return logs_folder

	@property
	def sampling_f(self) -> float:
		"""
		Returns the sampling frequency of the recording (in Hz).

		@return sampling_frequency: float
		"""

		return self.recording.get_sampling_frequency()

	@property
	def num_sortings(self) -> int:
		"""
		Returns the number of sortings.

		@return num_sortings: int
			The number of sortings in the LussacData object.
		"""

		return len(self.sortings)

	@staticmethod
	def _setup_probe(recording: si.BaseRecording, filename: str) -> si.BaseRecording:
		"""
		Loads the probe geometry into the 'recording' object.

		@param recording: BaseRecording
			Recording on which to load the probe geometry.
		@param filename: str
			Path to the JSON file containing the probe geometry (ProbeInterface format).
		@return probed_recording: BaseRecording
			The recording object with the probe geometry loaded.
		"""

		probe_group = probeinterface.io.read_probeinterface(filename)
		return recording.set_probegroup(probe_group)

	@staticmethod
	def _load_recording(params: dict) -> si.BaseRecording:
		"""
		Loads the recording from the given parameters.

		@param params: dict
			A dictionary containing Lussac's recording parameters.
		@return recording: BaseRecording
			The recording object.
		"""

		recording_extractor = se.extractorlist.get_recording_extractor_from_name(params['recording_extractor'])
		return recording_extractor(**params['extractor_params'])

	@staticmethod
	def _load_sortings(phy_folders: dict[str, str]) -> dict[str, se.PhySortingExtractor]:
		"""
		Loads all the sortings (in Phy format) and return

		@param phy_folders: dict[str, str]
			Dict containing the name as key, and the path to all the
			phy folder containing the spike sorted data as value.
		@return sortings: dict[str, PhySortingExtractor]
			Dictionary containing the Phy sorting objects indexed by their name.
		"""

		sortings = {}
		for name, path in phy_folders.items():
			sorting = se.PhySortingExtractor(path)
			sorting.annotate(name=name)
			sortings[name] = sorting

		return sortings

	@staticmethod
	def _format_params(params: dict[str, dict]) -> dict[str, dict]:
		"""
		Formats the parameters' dictionary to take care of semicolons ';'.
		When a semicolon appears in the category, it duplicates the module and
		runs it separately for all categories separated by the semicolon.

		@param params: dict
			The parameters' dictionary for the Lussac pipeline.
		@return formatted_params: dict:
			The formatted parameters' dictionary.
		"""

		formatted_params = {}

		for module, value in params.items():
			if not isinstance(module, str):
				raise Exception(f"Error: Module {module} in params['lussac']['pipeline'] must be a string.")
			if not isinstance(value, dict):
				raise Exception(f"Error: params['lussac']['pipeline][{module}] must map to a dict.")

			formatted_params[module] = {}

			for category, parameters in value.items():
				if not isinstance(category, str):
					raise Exception(f"Error: Category {category} in params['lussac']['pipeline'][{module}] must be a string.")
				if not isinstance(parameters, dict):
					raise Exception(f"Error: params['lussac']['pipeline'][{module}][{category}] must map to a dict.")

				categories = category.split(';')

				for cat in categories:
					formatted_params[module][cat] = copy.deepcopy(parameters)

		return formatted_params

	@staticmethod
	def _setup_tmp_directory(folder_path: str) -> tempfile.TemporaryDirectory:
		"""
		Created a directory for temporary files.
		This directory is deleted when Lussac exists (whether normally or because of a crash).

		@param folder_path: str
			Path for the temporary directory.
		@return temporary_directory: TemporaryDirectory
			The temporary directory object (from tempfile library).
		"""

		folder_path = pathlib.Path(folder_path)
		tmp_dir = tempfile.TemporaryDirectory(prefix=folder_path.name + '_', dir=folder_path.parent)

		os.mkdir(f"{tmp_dir.name}/spike_interface")

		return tmp_dir

	@staticmethod
	def create_from_params(params: dict[str, dict]) -> 'LussacData':
		"""
		Creates a new LussacData object from the given parameters.

		@param params: dict
			Lussac's parameters.
		@return: LussacData
			The newly created LussacData object.
		"""

		recording = LussacData._load_recording(params['recording'])
		recording = LussacData._setup_probe(recording, params['recording']['probe_file'])
		sortings = LussacData._load_sortings(params['phy_folders'])

		return LussacData(recording, sortings, params)


@dataclass(slots=True)
class MonoSortingData:
	"""
	Allows easy manipulation of the LussacData object when working on only one sorting.

	Attributes:
		data	The main data object for Lussac.
		sorting	The sorting being used.
	"""

	data: LussacData
	sorting: si.BaseSorting

	@property
	def recording(self) -> si.BaseRecording:
		"""
		Returns the recording object.

		@return recording: BaseRecording
			The recording object.
		"""

		return self.data.recording

	@property
	def name(self) -> str:
		"""
		Returns the name of the current active sorting.

		@return name: str
			The name of the current active sorting.
		"""

		return self.sorting.get_annotation("name")

	@property
	def sampling_f(self) -> float:
		"""
		Returns the sampling frequency of the recording (in Hz).

		@return sampling_frequency: float
			The sampling frequency (in Hz).
		"""

		return self.data.sampling_f

	@property
	def tmp_folder(self) -> str:
		"""
		Returns the path to the temporary folder.
		This folder is deleted when Lussac exists (normally or because of a crash).

		@return tmp_folder: str
			Path to the folder that store temporary files.
		"""

		return self.data.tmp_folder

	def get_unit_spike_train(self, unit_id: int) -> npt.NDArray[np.integer]:
		"""
		Returns the spike_train (i.e. an array containing all the spike timings)
		of a given unit.

		@param unit_id: int
			The cluster's ID of which to return the spike train.
		@return spike_train: np.ndarray
		"""

		return self.sorting.get_unit_spike_train(unit_id)
