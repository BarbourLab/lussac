import os
import pathlib
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
	sortings: list[si.BaseSorting]
	params: dict
	_tmp_directory: tempfile.TemporaryDirectory

	def __init__(self, params: dict) -> None:
		"""
		Creates a new LussacData instance.
		Loads all the necessary information from spike sorters output.

		@param params: dict
			The params.json file containing everything we need to know.
		"""

		self.params = params

		self.recording = self._load_recording(params['recording'])
		self._setup_probe(params['recording']['probe_file'])

		self.sortings = self._load_sortings(params['phy_folders'])
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

	def _setup_probe(self, filename: str) -> None:
		"""
		Loads the probe geometry into the 'recording' attribute.

		@param filename: str
			Path to the JSON file containing the probe geometry (ProbeInterface format).
		"""

		probe_group = probeinterface.io.read_probeinterface(filename)
		self.recording = self.recording.set_probegroup(probe_group)

	@staticmethod
	def _load_recording(params: dict) -> si.BaseRecording:
		"""
		Loads the recording from the given parameters.

		@param params: dict
			A dictionary containing Lussac's recording parameters.
		@return recording: BaseRecording
			The recording object.
		"""

		recording_extractor = None
		for extractor in se.recording_extractor_full_list:
			if extractor.__name__ == params['recording_extractor']:
				recording_extractor = extractor
				break

		if recording_extractor is None:
			raise ValueError(f"Recording extractor '{params['recording_extractor']}' not found.")

		return recording_extractor(params['file'], **params['extractor_params'])

	@staticmethod
	def _load_sortings(phy_folders: dict[str, str]) -> list[se.PhySortingExtractor]:
		"""
		Loads all the sortings (in Phy format) and return

		@param phy_folders: dict[str, str]
			Dict containing the name as key, and the path to all the
			phy folder containing the spike sorted data as value.
		@return sortings: list[PhySortingExtractor]
			List containing the Phy sorting objects.
		"""

		sortings = []
		for name, path in phy_folders.items():
			sorting = se.PhySortingExtractor(path)
			sorting.annotate(name=name)
			sortings.append(sorting)

		return sortings

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


@dataclass(slots=True)
class MonoSortingData:
	"""
	Allows easy manipulation of the LussacData object when working on only one sorting.

	Attributes:
		data			The main data object for Lussac.
		active_sorting	Which sorting is currently being used?
	"""

	data: LussacData
	active_sorting: int

	@property
	def recording(self) -> si.BaseRecording:
		"""
		Returns the recording object.

		@return recording: BaseRecording
			The recording object.
		"""

		return self.data.recording

	@property
	def sorting(self) -> si.BaseSorting:
		"""
		Returns the current active sorting.

		@return sorting: BaseSorting
			The current active sorting.
		"""

		return self.data.sortings[self.active_sorting]

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