import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pathlib
import shutil
from typing import Any
import numpy as np
from lussac.core import MonoSortingData, MultiSortingsData
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.qualitymetrics as sqm


@dataclass(slots=True)
class LussacModule(ABC):
	"""
	The abstract Module class.
	Every module used in Lussac must inherit from this class.

	Attributes:
		name			Module's name (i.e. the key in the pipeline dictionary).
		data 			Reference to the data object.
		category		What category is used for the module (i.e. the key in the dictionary).
		export_sortings	Whether to export the sortings after the module is executed.
	"""

	name: str
	data: MonoSortingData | MultiSortingsData
	category: str
	export_sortings: bool = True

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

	def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
		"""
		Updates the parameters with the default parameters (if there is a conflict, the given parameters are taken).
		Need to update recursively because the parameters are usually a nested dictionary.
		This is done by flattening then unflattening the dictionaries.

		@param params: dict[str, Any]
			The parameters to update.
		@return updated_params: dict[str, Any]
			The parameters updated with the module's default parameters.
		"""

		return utils.merge_dict(params, self.default_params)

	def create_analyzer(self, sorting: si.BaseSorting, sub_folder: str | None = None, filter_band: list[float, float] | None = None, **params) -> si.SortingAnalyzer:
		"""
		Creates the SortingAnalyzer object and returns it.

		@param sorting: BaseSorting
			The sorting for the SortingAnalyzer.
		@param sub_folder: str | None:
			The sub-folder where to save the waveforms.
		@param params
			The parameters for the sorting analyzer.
		@param filter_band: list[float, float] | None
			The cutoff frequencies for the Gaussian bandpass filter to apply to the recording.
		@return analyzer: SortingAnalyzer
			The sorting analyzer object.
		"""
		if sub_folder is None:
			sub_folder = "analyzer"

		folder_path = f"{self.data.tmp_folder}/{self.name}/{self.category}/{sorting.get_annotation('name')}/{sub_folder}"

		recording = self.recording
		if filter_band is not None:
			assert len(filter_band) == 2, "The filter must be a list of 2 elements [min_cutoff, max_cutoff] (in Hz)."
			recording = spre.gaussian_filter(recording, *filter_band, margin_sd=2)

		sorting = sorting.to_numpy_sorting()  # Convert sorting for faster extraction.
		if pathlib.Path(folder_path).exists():
			return si.load_sorting_analyzer(folder=folder_path)
		else:
			return si.create_sorting_analyzer(sorting, recording, format="binary_folder", folder=folder_path, **params)


@dataclass(slots=True)
class MonoSortingModule(LussacModule):
	"""
	The abstract mono-sorting module class.
	This is for modules that don't work on multiple sortings at once.

	Attributes:
		name			Module's name (i.e. the key in the pipeline dictionary).
		data			Reference to the mono-sorting data object.
		category		What category is used for the module (i.e. the key in the dictionary).
		export_sortings	Whether to export the sortings after the module is executed.
	"""

	data: MonoSortingData

	def __del__(self) -> None:
		"""
		When the module is garbage collected, remove the temporary folder.
		"""

		if (self.data.tmp_folder / self.name).exists():
			shutil.rmtree(self.data.tmp_folder / self.name)

	@property
	def sorting(self) -> si.BaseSorting:
		"""
		Returns the sorting object.

		@return sorting: BaseSorting
			The sorting object.
		"""

		return self.data.sorting

	@property
	def logs_folder(self) -> pathlib.Path:
		"""
		Returns the logs directory for this module.

		@return logs_folder: Path
			Path to the logs directory.
		"""

		logs_folder = self.data.logs_folder / self.name / self.category / self.data.name
		logs_folder.mkdir(parents=True, exist_ok=True)

		return logs_folder

	@property
	def tmp_folder(self) -> pathlib.Path:
		"""
		Returns the temporary directory for this module.

		@return: tmp_folder: Path
			Path to the temporary directory.
		"""

		tmp_folder = self.data.tmp_folder / self.name / self.category / self.data.name
		tmp_folder.mkdir(parents=True, exist_ok=True)

		return tmp_folder

	@abstractmethod
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		...

	def create_analyzer(self, sorting: si.BaseSorting | None = None, sub_folder: str | None = None, filter_band: list[float, float] | None = None, **params) -> si.SortingAnalyzer:
		"""
		Calls the parent LussacModule.create_analyzer
		'sorting' argument is optional. If None (default), will take the MonoSortingModule.data.sorting
		"""

		if sorting is None:
			sorting = self.sorting

		return super(MonoSortingModule, self).create_analyzer(sorting, sub_folder, filter_band, **params)

	def get_templates(self, max_spikes_per_unit: int = 1000, ms_before: float = 1.0, ms_after: float = 3.0, filter_band: tuple[float] | list[float] | np.ndarray | None = None,
					  margin: float = 3.0, sub_folder: str = "templates", return_analyzer: bool = False) -> np.ndarray | tuple[np.ndarray, si.SortingAnalyzer, int]:
		"""
		Extract the templates for all the units.
		If filter_band is not None, will also filter them using a Gaussian filter.

		@param max_spikes_per_unit: int
			The maximum number of spikes to extract per unit to evaluate the templates.
		@param ms_before: float
			The time before the spike (in ms) to extract.
		@param ms_after: float
			The time after the spike (in ms) to extract.
		@param filter_band: Iterable[float, float] | None
			If not none, the highpass and lowpass cutoff frequencies for Gaussian filtering (in Hz).
		@param margin: float
			The margin (in ms) to extract (useful for filtering).
		@param sub_folder: str
			The sub-folder used for the sorting analyzer.
		@param return_analyzer: bool
			If true, will also return the sorting analyzer and margin (in samples).
		@return templates: np.ndarray (n_units, n_samples, n_channels)
			The extracted templates
		@return analyzer: si.SortingAnalyzer
			The sorting analyzer of unfiltered waveforms.
			Only if return_analyzer is True.
		@return margin: int
			The margin (in samples) that were used for the filtering.
			Only if return_analyzer is True.
		"""

		ms_before += margin
		ms_after += margin
		analyzer = self.create_analyzer(sub_folder=sub_folder)
		analyzer.compute({
			'random_spikes': {'max_spikes_per_unit': max_spikes_per_unit},
			'fast_templates': {'ms_before': ms_before, 'ms_after': ms_after}
		})
		templates = analyzer.get_extension("fast_templates").get_data()

		if filter_band is not None:
			templates = utils.filter(templates, filter_band, axis=1)

		margin = int(round(margin * self.recording.sampling_frequency * 1e-3))

		if return_analyzer:
			return templates[:, margin:-margin], analyzer, margin
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
		params = copy.deepcopy(params)
		default_params = {
			'firing_rate': {},
			'contamination': {},
			'amplitude': {
				'wvf_extraction': {'ms_before': 1.0, 'ms_after': 1.0, 'max_spikes_per_unit': 500},
				'peak_sign': "both",
				'mode': "extremum",
				'filter': [100, 9_000]
			},
			'SNR': {
				'wvf_extraction': {'ms_before': 1.0, 'ms_after': 1.0, 'max_spikes_per_unit': 500},
				'peak_sign': "both",
				'mode': "extremum",
				'filter': [100, 9_000]
			},
			'sd_ratio': {
				'wvf_extraction': {'ms_before': 1.0, 'ms_after': 1.0, 'max_spikes_per_unit': 500},
				'spike_amplitudes_kwargs': {'peak_sign': "both"},
				'sd_ratio_kwargs': {},
				'filter': [100, 9_000]
			},
			'ISI_portion': {}
		}

		if attribute not in default_params:
			raise ValueError(f"Unknown attribute '{attribute}'.")
		params = utils.merge_dict(params, default_params[attribute])

		recording = self.data.recording
		sorting = self.sorting

		filter_band = None if 'filter' not in params else params['filter']

		analyzer = self.create_analyzer(sub_folder=attribute, filter_band=filter_band)

		match attribute:
			case "firing_rate":  # Returns the firing rate of each unit (in Hz).
				n_spikes = {unit_id: len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids}
				firing_rates = {unit_id: n_spike * sorting.get_sampling_frequency() / recording.get_num_frames() for unit_id, n_spike in n_spikes.items()}
				return firing_rates

			case "contamination":  # Returns the estimated contamination of each unit.
				censored_period, refractory_period = params['refractory_period']
				contamination, _ = sqm.compute_refrac_period_violations(analyzer, refractory_period_ms=refractory_period, censored_period_ms=censored_period)
				return contamination

			case "amplitude":  # Returns the amplitude of each unit on its best channel (unit depends on the wvf extractor 'return_scaled' parameter).
				analyzer.compute({
					'random_spikes': {'max_spikes_per_unit': params['wvf_extraction']['max_spikes_per_unit']},
					'fast_templates': {'ms_before': params['wvf_extraction']['ms_before'], 'ms_after': params['wvf_extraction']['ms_after']}
				})
				params = utils.filter_kwargs(params, si.template_tools.get_template_extremum_amplitude)
				amplitudes = si.get_template_extremum_amplitude(analyzer, **params)
				return amplitudes

			case "SNR":  # Returns the signal-to-noise ratio of each unit on its best channel.
				analyzer.compute({
					'noise_levels': {},
					'random_spikes': {'max_spikes_per_unit': params['wvf_extraction']['max_spikes_per_unit']},
					'fast_templates': {'ms_before': params['wvf_extraction']['ms_before'], 'ms_after': params['wvf_extraction']['ms_after']}
				})
				params = utils.filter_kwargs(params, sqm.compute_snrs)
				SNRs = sqm.compute_snrs(analyzer, **params)
				return SNRs

			case "sd_ratio":  # Returns the standard deviation of the amplitude of spikes divided by the standard deviation on the same channel.
				analyzer.compute({
					'random_spikes': {'max_spikes_per_unit': params['wvf_extraction']['max_spikes_per_unit']},
					'fast_templates': {'ms_before': params['wvf_extraction']['ms_before'], 'ms_after': params['wvf_extraction']['ms_after']},
					'spike_amplitudes': params['spike_amplitudes_kwargs']
				})
				sd_ratio = sqm.compute_sd_ratio(analyzer, **params['sd_ratio_kwargs'])
				return sd_ratio

			case "ISI_portion":  # Returns the portion of consecutive spikes that are between a certain range (in ms).
				low, high = np.array(params['range']) * recording.sampling_frequency * 1e-3
				diff = {unit_id: np.diff(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids}
				ISI_portion = {unit_id: np.nan if len(d) == 0 else (np.sum((low < d) & (d < high)) / len(d)) for unit_id, d in diff.items()}
				return ISI_portion

			case _:  # pragma: no cover (unreachable code)
				raise ValueError(f"Unknown attribute: '{attribute}'")

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
		name			Module's name (i.e. the key in the pipeline dictionary).
		data			Reference to Lussac data object.
		category		What category is used for the module (i.e. the key in the dictionary).
		export_sortings	Whether to export the sortings after the module is executed.
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
	def logs_folder(self) -> pathlib.Path:
		"""
		Returns the logs directory for this module.

		@return logs_folder: Path
			Path to the logs directory.
		"""

		logs_folder = self.data.logs_folder / self.name / self.category
		logs_folder.mkdir(parents=True, exist_ok=True)

		return logs_folder

	@property
	def tmp_folder(self) -> pathlib.Path:
		"""
		Returns the temporary directory for this module.

		@return tmp_folder: Path
			Path to the temporary directory.
		"""

		tmp_folder = self.data.tmp_folder / self.name / self.category
		tmp_folder.mkdir(parents=True, exist_ok=True)

		return tmp_folder

	@abstractmethod
	def run(self, params: dict[str, Any]) -> dict[str, si.BaseSorting]:
		...

	def create_analyzer(self, sub_folder: str | None = None, filter_band: list[float, float] | None = None, **params) -> si.SortingAnalyzer:
		"""
		Aggregates all sortings and calls parent LussacModule.create_analyzer.
		The returned SortingAnalyzer has a variable 'renamed_unit_ids' which is a dict[str, dict[Any, Any]]
		where the first key is the analysis name, and the second one is the 'old' unit_id.
		"""

		aggregated_sortings = si.aggregate_units(list(self.sortings.values()))
		aggregated_sortings.annotate(name="aggregated_sortings")
		analyzer = super(MultiSortingsModule, self).create_analyzer(aggregated_sortings, sub_folder, filter_band, **params)

		analyzer.sortings = self.sortings
		analyzer.renamed_unit_ids = {}
		renamed_unit_id = 0
		for sorting_name in self.sortings.keys():
			analyzer.renamed_unit_ids[sorting_name] = {}

			for unit_id in self.sortings[sorting_name].unit_ids:
				analyzer.renamed_unit_ids[sorting_name][unit_id] = renamed_unit_id
				renamed_unit_id += 1

		return analyzer
