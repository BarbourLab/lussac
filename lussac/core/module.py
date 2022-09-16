from abc import ABC, abstractmethod
from dataclasses import dataclass
import spikeinterface.core as si
from lussac.core.lussac_data import LussacData, MonoSortingData


@dataclass(slots=True)
class LussacModule(ABC):
	"""
	The abstract Module class.
	Every module used in Lussac must inherit from this class.

	Attributes:
		name		Module's name (i.e. the key in the pipeline dictionary).
		data 		Reference to the data object.
		category	What category is used for the module (i.e. the key in the dictionary).
		logs_folder	Path to the folder where to output the logs.
	"""

	name: str
	data: object
	category: str
	logs_folder: str


@dataclass(slots=True)
class MonoSortingModule(LussacModule):
	"""
	The abstract mono-sorting module class.
	This is for modules that don't work on multiple sortings at once.

	Attributes:
		name		Module's name (i.e. the key in the pipeline dictionary).
		data		Reference to the mono-sorting data object.
		logs_folder	Path to the folder where to output the logs.
	"""

	data: MonoSortingData

	@abstractmethod
	def run(self, params: dict) -> si.BaseSorting:
		...

	def extract_waveforms(self, sub_folder: str = None, **params) -> si.WaveformExtractor:
		"""
		Creates the WaveformExtractor object and returns it.

		@param sub_folder: str:
			The sub-folder where to save the waveforms.
		@param params
			The parameters for the waveform extractor.
		@return wvf_extractor: WaveformExtractor
			The waveform extractor object.
		"""

		folder_path = f"{self.data.tmp_folder}/{self.name}/waveforms/{self.category}/{self.data.name}"
		if sub_folder is not None:
			folder_path += f"/{sub_folder}"

		return si.extract_waveforms(self.data.recording, self.data.sorting, folder_path, **params)


@dataclass(slots=True)
class MultiSortingsModule(LussacModule):
	"""
	The abstract multi-sorting module class.
	This is for modules that work on multiple sortings at once.

	Attributes:
		name		Module's name (i.e. the key in the pipeline dictionary).
		data		Reference to Lussac data object.
		logs_folder	Path to the folder where to output the logs.
	"""

	data: LussacData

	@abstractmethod
	def run(self):
		...
