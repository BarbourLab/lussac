import copy
import datetime
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
import numba
import numpy as np
import numpy.typing as npt
from plotly.offline.offline import get_plotlyjs
from lussac.utils import Utils
import probeinterface.io
import spikeinterface.core as si
import spikeinterface.curation as scur
import spikeinterface.extractors as se


class LussacData:
	"""
	The main data object for Lussac.

	Attributes:
		recording		A SpikeInterface recording object.
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
		@param sortings: dict[str, BaseSorting]
			A dict containing all the sorting objects (i.e. all the analyses).
			The keys are the analyses' names.
		@param params: dict
			The params.json file containing everything we need to know.
		"""

		self.recording = recording
		self.sortings = {name: scur.remove_excess_spikes(sorting.remove_empty_units(), recording) for name, sorting in sortings.items()}
		params['lussac']['pipeline'] = self._format_params(params['lussac']['pipeline'])
		self.params = params
		self._tmp_directory = self._setup_tmp_directory(params['lussac']['tmp_folder'])
		self._setup_logs_directory(params['lussac']['logs_folder'], params['lussac']['overwrite_logs'])

		if 'si_global_job_kwargs' in params['lussac']:
			si.set_global_job_kwargs(**params['lussac']['si_global_job_kwargs'])
			numba.set_num_threads(si.get_global_job_kwargs()['n_jobs'])

		targets = logging.StreamHandler(sys.stdout), logging.FileHandler(self.logs_folder / "lussac.logs")
		targets[0].terminator = ''
		targets[1].terminator = ''
		logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=targets)
		logging.info(f"\nRunning Lussac!\n{datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

		self._sanity_check()

		Utils.sampling_frequency = recording.sampling_frequency
		Utils.t_max = recording.get_num_frames()

	def clone(self) -> 'LussacData':
		"""
		Returns a copy of the LussacData object.

		@return: LussacData
			A copy of the LussacData object.
		"""

		data = copy.copy(self)
		data.recording = data.recording.clone()
		data.sortings = {name: sorting.clone() for name, sorting in data.sortings.items()}
		data.params = copy.deepcopy(self.params)

		return data

	@property
	def tmp_folder(self) -> pathlib.Path:
		"""
		Returns the path to the temporary folder.
		This folder is deleted when Lussac exists (normally or because of a crash).

		@return tmp_folder: Path
			Path to the folder that store temporary files.
		"""

		return pathlib.Path(self._tmp_directory.name)

	@property
	def logs_folder(self) -> pathlib.Path:
		"""
		Returns the path to the logs folder.

		@return: logs_folder: Path
			Path to the folder that stores the logs.
		"""

		return pathlib.Path(self.params['lussac']['logs_folder'])

	@property
	def sampling_f(self) -> float:
		"""
		Returns the sampling frequency of the recording (in Hz).

		@return sampling_frequency: float
		"""

		return self.recording.sampling_frequency

	@property
	def num_sortings(self) -> int:
		"""
		Returns the number of sortings.

		@return num_sortings: int
			The number of sortings in the LussacData object.
		"""

		return len(self.sortings)

	def _sanity_check(self) -> None:
		"""
		Checks that everything seems correct in the recording and sortings.
		"""

		assert self.recording.get_num_frames() > 0
		assert self.recording.get_num_channels() > 0
		assert self.recording.get_probegroup() is not None

		for name, sorting in self.sortings.items():
			assert sorting.get_sampling_frequency() == self.sampling_f
			assert sorting.get_num_segments() == self.recording.get_num_segments()
			assert sorting.get_annotation("name") == name

			# Check that spike trains are valid.
			spike_vector = sorting.to_spike_vector()
			assert spike_vector['sample_index'][0] >= 0
			assert spike_vector['sample_index'][-1] < self.recording.get_num_frames()
			assert np.all(np.diff(spike_vector['sample_index']) >= 0)

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
	def _load_sortings(sortings_path: dict[str, str]) -> dict[str, si.BaseSorting]:
		"""
		Loads all the sortings (in Phy format) and return

		@param sortings_path: dict[str, str]
			Dict containing the name as key, and the path to all the
			phy folder / SpikeInterface extractor containing the spike sorted data as value.
		@return sortings: dict[str, BaseSorting]
			Dictionary containing the Phy sorting objects indexed by their name.
		"""

		sortings = {}
		for name, path in sortings_path.items():
			path = pathlib.Path(path)

			if not path.exists():
				raise FileNotFoundError(f"Could not find the sorting file {path}.")
			elif path.is_dir() and (path / "spike_times.npy").exists():
				sorting = se.PhySortingExtractor(path)
			else:
				sorting = si.load_extractor(path, base_folder=True)
				assert isinstance(sorting, si.BaseSorting)

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
				raise Exception(f"Error: params['lussac']['pipeline'][{module}] must map to a dict.")

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
		folder_path.parent.mkdir(parents=True, exist_ok=True)
		tmp_dir = tempfile.TemporaryDirectory(prefix=folder_path.name + '_', dir=folder_path.parent)

		return tmp_dir

	@staticmethod
	def _setup_logs_directory(folder_path: str, overwrite_logs: bool = False) -> None:
		"""
		Sets up the folder containing Lussac's logs.

		@param folder_path: str
			Path for the logs directory.
		@param overwrite_logs: bool
			If True, will erase the logs folder if it already exists.
		"""

		if overwrite_logs and os.path.exists(folder_path):
			shutil.rmtree(folder_path)

		os.makedirs(folder_path, exist_ok=True)

		# The javascript for plotly html files is about ~3 MB.
		# To not export it for each plot, store it in the logs and point to it in each html file.
		Utils.plotly_js_file = pathlib.Path(f"{folder_path}/plotly.min.js")
		if not os.path.exists(Utils.plotly_js_file):
			file = open(Utils.plotly_js_file, 'w+', encoding="utf-8")
			file.write(get_plotlyjs())
			file.close()

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
		if 'probe_file' in params['recording']:
			recording = LussacData._setup_probe(recording, str(pathlib.Path(params['recording']['probe_file']).absolute()))
		sortings = LussacData._load_sortings(params['analyses'] if 'analyses' in params else {})

		return LussacData(recording, sortings, params)


@dataclass(slots=True)
class MonoSortingData:
	"""
	Allows easy manipulation of the LussacData object when working on only one sorting (or sub-sorting).

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
	def tmp_folder(self) -> pathlib.Path:
		"""
		Returns the path to the temporary folder.
		This folder is deleted when Lussac exists (normally or because of a crash).

		@return tmp_folder: Path
			Path to the folder that store temporary files.
		"""

		return self.data.tmp_folder

	@property
	def logs_folder(self) -> pathlib.Path:
		"""
		Returns the path to the logs folder.

		@return logs_folder: Path
			Path to the folder that stores Lussac's logs.
		"""

		return self.data.logs_folder

	def get_unit_spike_train(self, unit_id: int) -> npt.NDArray[np.integer]:
		"""
		Returns the spike_train (i.e. an array containing all the spike timings)
		of a given unit.

		@param unit_id: int
			The cluster's ID of which to return the spike train.
		@return spike_train: np.ndarray
		"""

		return self.sorting.get_unit_spike_train(unit_id)


@dataclass(slots=True)
class MultiSortingsData:
	"""
	Allows easy manipulation of the LussacData object when working on multiple sortings (or sub-sortings).

	Attributes:
		data		The main data object for Lussac.
		sortings	The sortings being used.
	"""

	data: LussacData
	sortings: dict[str, si.BaseSorting]

	@property
	def recording(self) -> si.BaseRecording:
		"""
		Returns the recording object.

		@return recording: BaseRecording
			The recording object.
		"""

		return self.data.recording

	@property
	def num_sortings(self) -> int:
		"""
		Returns the number of sortings.

		@return num_sortings: int
			The number of sortings.
		"""

		return len(self.sortings)

	@property
	def tmp_folder(self) -> pathlib.Path:
		"""
		Returns the path to the temporary folder.
		This folder is deleted when Lussac exists (normally or because of a crash).

		@return tmp_folder: Path
			Path to the folder that store temporary files.
		"""

		return self.data.tmp_folder

	@property
	def logs_folder(self) -> pathlib.Path:
		"""
		Returns the path to the logs folder.

		@return logs_folder: Path
			Path to the folder that stores Lussac's logs.
		"""

		return self.data.logs_folder
