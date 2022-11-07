from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any
import numpy as np
from lussac.core.lussac_data import MonoSortingData, MultiSortingsData
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm


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
	data: MonoSortingData | MultiSortingsData
	category: str

	@property
	def recording(self) -> si.BaseRecording:
		"""
		Returns the recording object.

		@return recording: BaseRecording
			The recording object.
		"""

		return self.data.recording

	@property
	def sampling_f(self) -> float:
		"""
		Returns the sampling frequency of the recording (in Hz).

		@return sampling_f: float
			The sampling frequency (in Hz).
		"""

		return self.recording.sampling_frequency

	@abstractmethod
	def run(self, params: dict[str, Any]) -> si.BaseSorting | dict[str, si.BaseSorting]:
		"""
		Executes the module and returns the result (either a sorting of a dict of sortings).

		@param params: dict
			The parameters for the module.
		@return result: si.BaseSorting | dict[str, si.BaseSorting]
			The result of the module.
		"""
		...

	@property
	@abstractmethod
	def default_params(self) -> dict[str, Any]:
		"""
		Returns the default parameters of the module.

		@return default_params: dict[str, Any]
			The default parameters of the module.
		"""
		...


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

	@property
	def sorting(self) -> si.BaseSorting:
		"""
		Returns the sorting object.

		@return sorting: BaseSorting
			The sorting object.
		"""

		return self.data.sorting

	@property
	def logs_folder(self) -> str:
		"""
		Returns the logs directory for this module.

		@return logs_folder: str
			Path to the logs directory.
		"""

		logs_folder = f"{self.data.logs_folder}/{self.name}/{self.category}/{self.data.name}"
		if not os.path.exists(logs_folder):
			os.makedirs(logs_folder)

		return logs_folder

	@abstractmethod
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		...

	def extract_waveforms(self, sorting: si.BaseSorting | None = None, sub_folder: str | None = None, **params) -> si.WaveformExtractor:
		"""
		Creates the WaveformExtractor object and returns it.

		@param sorting: BaseSorting | None
			The sorting for the WaveformExtractor.
			If None, will take the sorting from the data object.
		@param sub_folder: str | None:
			The sub-folder where to save the waveforms.
		@param params
			The parameters for the waveform extractor.
		@return wvf_extractor: WaveformExtractor
			The waveform extractor object.
		"""

		folder_path = f"{self.data.tmp_folder}/{self.name}/{self.category}/{self.data.name}"
		if sub_folder is not None:
			folder_path += f"/{sub_folder}"
		folder_path += f"/wvf_extractor"

		sorting = self.sorting if sorting is None else sorting
		return si.extract_waveforms(self.data.recording, sorting, folder_path, allow_unfiltered=True, **params)

	def get_templates(self, params: dict, filter_band: tuple[float, float] | list[float, float] | np.ndarray | None = None, margin: float = 3.0,
					  return_extractor: bool = False) -> np.ndarray | tuple[np.ndarray, si.WaveformExtractor, int]:
		"""
		Extract the templates for all the units.
		If filter_band is not None, will also filter them using a Gaussian filter.

		@param params: dict
			The parameters for the waveform extraction.
		@param filter_band: Iterable[float, float] | None
			If not none, the highpass and lowpass cutoff frequencies (in Hz).
		@param margin: float
			The margin (in ms) to extract (useful for filtering).
		@param return_extractor: bool
			If true, will also return the waveform extractor and margin (in samples).
		@return templates: np.ndarray (n_units, n_samples, n_channels)
			The extracted templates
		@return wvf_extractor: si.WaveformExtractor
			The Waveform Extractor of unfiltered waveforms.
			Only if return_extractor is True.
		@return margin: int
			The margin (in samples) that were used for the filtering.
			Only if return_extractor is True.
		"""

		params['ms_before'] += margin
		params['ms_after'] += margin
		wvf_extractor = self.extract_waveforms(sub_folder="templates", **params)
		templates = wvf_extractor.get_all_templates()

		if filter_band is not None:
			templates = utils.filter(templates, filter_band, axis=1)

		margin = int(round(margin * self.recording.sampling_frequency * 1e-3))

		if return_extractor:
			return templates[:, margin:-margin], wvf_extractor, margin
		else:
			return templates[:, margin:-margin]

	def get_units_attribute(self, attribute: str, params: dict) -> dict:
		"""
		Gets the attribute for all the units.

		@param attribute: str
			The attribute name.
			- firing_rate (in Hz)
			- contamination (between 0 and 1)
			- amplitude (unit depends on the wvf extractor 'return_scaled' parameter)
			- amplitude_std (unit depends on parameters 'return_scaled')
		@param params: dict
			The parameters to get the attribute.
			- 'filter': parameters to filter the recording.
			- 'wvf_extraction': parameters to extract the waveforms.
			- others: parameters for how to get the attribute.
		@return attribute: np.ndarray
			The attribute for all the units.
		"""
		recording = self.data.recording
		sorting = self.sorting
		if 'filter' in params:
			recording = spre.filter(recording, **params['filter'])

		wvf_extractor = self.extract_waveforms(sub_folder=attribute, **params['wvf_extraction']) if 'wvf_extraction' in params \
						else si.WaveformExtractor(recording, sorting, allow_unfiltered=True)

		# TODO: Probably a better way to handle 'params' than manually setting each parameter individually.
		match attribute:
			case "firing_rate":  # Returns the firing rate of each unit (in Hz).
				n_spikes = {unit_id: len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids}
				firing_rates = {unit_id: n_spike * sorting.get_sampling_frequency() / recording.get_num_frames() for unit_id, n_spike in n_spikes.items()}
				return firing_rates

			case "contamination":  # Returns the estimated contamination of each unit.
				censored_period, refractory_period = params['refractory_period']
				contamination = sqm.compute_refrac_period_violations(wvf_extractor, refractory_period, censored_period)[1]
				return contamination

			case "amplitude":  # Returns the amplitude of each unit on its best channel (unit depends on the wvf extractor 'return_scaled' parameter).
				peak_sign = params['peak_sign'] if 'peak_sign' in params else "both"
				mode = params['mode'] if 'mode' in params else "extremum"
				amplitudes = spost.get_template_extremum_amplitude(wvf_extractor, peak_sign, mode)
				return amplitudes

			case "amplitude_std":  # Returns the standard deviation of the amplitude of spikes.
				peak_sign = params['peak_sign'] if 'peak_sign' in params else "both"
				return_scaled = params['return_scaled'] if 'return_scaled' in params else True
				chunk_duration = params['chunk_duration'] if 'chunk_duration' in params else '1s'
				n_jobs = params['n_jobs'] if 'n_jobs' in params else 6
				amplitudes = spost.compute_spike_amplitudes(wvf_extractor, peak_sign=peak_sign, return_scaled=return_scaled, outputs='by_unit', chunk_duration=chunk_duration, n_jobs=n_jobs)[0]
				std_amplitudes = {unit_id: np.std(amp) for unit_id, amp in amplitudes.items()}
				return std_amplitudes

			case "ISI_portion":  # Returns the portion of consecutive spikes that are between a certain range (in ms).
				low, high = np.array(params['range']) * recording.sampling_frequency * 1e-3
				diff = {unit_id: np.diff(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids}
				ISI_portion = {unit_id: np.sum((low < d) & (d < high)) / len(d) for unit_id, d in diff.items()}
				return ISI_portion

			case _:
				raise ValueError(f"Unknown attribute: {attribute}")

	def get_units_attribute_arr(self, attribute: str, params: dict) -> np.array:
		"""
		See MonoSortingModule.get_units_attribute.
		Returns the same value but as a numpy array rather than a dict.
		"""

		return np.array(list(self.get_units_attribute(attribute, params).values()))


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

	data: MultiSortingsData

	@property
	def sortings(self) -> dict[str, si.BaseSorting]:
		"""
		Returns the sorting objects.

		@return sortings: dict[str, BaseSorting]
			The sorting objects.
		"""

		return self.data.sortings

	@property
	def logs_folder(self) -> str:
		"""
		Returns the logs directory for this module.

		@return logs_folder: str
			Path to the logs directory.
		"""

		logs_folder = f"{self.data.logs_folder}/{self.name}/{self.category}"
		if not os.path.exists(logs_folder):
			os.makedirs(logs_folder)

		return logs_folder

	@abstractmethod
	def run(self, params: dict[str, Any]) -> dict[str, si.BaseSorting]:
		...
