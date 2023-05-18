import math
import pathlib
from typing import Any
import numpy as np
import numpy.typing as npt
import spikeinterface.core as si


class TemplateExtractor:
	"""
	Lazy template extractor for fast computation.
	Only works for average

	Attributes
		recording	The recording extractor containing the voltage traces.
		sorting		The sorting extractor containing the spike times.
		params		The waveforms parameters for extraction.
		_templates	The templates memory map.
	"""

	__slots__ = "recording", "sorting", "folder", "params", "_templates"
	recording: si.BaseRecording
	sorting: si.BaseSorting
	folder: pathlib.Path
	params: dict[str, Any]
	_templates: np.memmap

	def __init__(self, recording: si.BaseRecording, sorting: si.BaseSorting, folder: pathlib.Path, params: dict[str, Any] | None = None,
				 templates_dtype: npt.DTypeLike = np.float32) -> None:
		"""
		Creates a new TemplateExtractor instance.
		Lazy extractor to extract only the necessary units on the necessary channels.

		@param recording: BaseRecording
			The recording object.
		@param sorting: BaseSorting
			The sorting object.
		@param folder: pathlib.Path
			The folder where the waveforms/templates are saved.
			Waveforms will be deleted once the templates are computed and saved.
		@param params: dict[str, Any] | None
			The waveforms parameters for extraction.
			If None, will take default values.
		@param templates_dtype: DTypeLike
			The dtype of the templates (by default: np.float32).
			Must support np.nan.
		"""

		self.recording = recording
		self.sorting = sorting
		self.folder = folder
		if params is None:
			params = {}
		self.set_params(**params)

		folder.mkdir(parents=True, exist_ok=True)
		self._templates = np.memmap(str(folder / "templates.npy"), dtype=templates_dtype, mode='w+', shape=(self.num_units, self.nsamples, self.num_channels))
		self._templates[:] = np.nan

	@property
	def sampling_frequency(self) -> float:
		"""
		Returns the sampling frequency of the recording.

		@return sampling_frequency: float
			The sampling frequency of the recording.
		"""

		return self.recording.sampling_frequency

	@property
	def unit_ids(self):
		"""
		Returns the unit ids of the sorting.

		@return unit_ids:
			The unit ids of the sorting.
		"""

		return self.sorting.unit_ids

	@property
	def num_units(self) -> int:
		"""
		Returns the number of units in the sorting.

		@return num_units: int
			The number of units in the sorting.
		"""

		return len(self.unit_ids)

	@property
	def channel_ids(self):
		"""
		Returns the channel ids of the recording.

		@return channel_ids:
			The channel ids of the recording.
		"""

		return self.recording.channel_ids

	@property
	def num_channels(self) -> int:
		"""
		Returns the number of channels in the recording.

		@return num_channels: int
			The number of channels in the recording.
		"""

		return len(self.channel_ids)

	@property
	def name(self) -> str:
		"""
		Returns the analysis' name for this TemplateExtractor.

		@return name: str
			The analysis' name.
		"""

		return self.sorting.get_annotation("name")

	@property
	def nbefore(self) -> int:
		"""
		Returns the number of samples before the spike time.

		@return nbefore: int
			The number of samples before the spike time.
		"""

		return math.ceil(self.params['ms_before'] * self.sampling_frequency * 1e-3)

	@property
	def nafter(self) -> int:
		"""
		Returns the number of samples after the spike time.

		@return nafter: int
			The number of samples after the spike time.
		"""

		return math.ceil(self.params['ms_after'] * self.sampling_frequency * 1e-3)

	@property
	def nsamples(self) -> int:
		"""
		Returns the total number of samples in time.

		@return nsamples: int
			The total number of samples.
		"""

		return self.nbefore + 1 + self.nafter

	def set_params(self, ms_before: float = 1.0, ms_after: float = 2.0, max_spikes_per_unit: int | None = 1_000, max_spikes_sparsity: int = 100) -> None:
		self.params = {
			'ms_before': ms_before,
			'ms_after': ms_after,
			'max_spikes_per_unit': max_spikes_per_unit,
			'max_spikes_sparsity': max_spikes_sparsity
		}

	def get_template(self, unit_id, channel_ids, return_scaled: bool = False) -> np.ndarray:
		"""
		Returns the template for a given unit and channels.
		If not computed, will compute it on the fly.
		Returns a copy array.

		@param unit_id:
			The unit id for which to return the template.
		@param channel_ids:
			The channel ids for which to return the template.
		@param return_scaled: bool
			If True, will return the templates scaled to µV (default: False).
		@return template: array (n_samples_time, n_channels)
			The template for the given unit and channels (as a copy).
		"""

		if channel_ids is None:
			channel_ids = self.channel_ids

		channel_indices = self.recording.ids_to_indices(channel_ids)
		template = self._templates[unit_id][:, channel_indices]

		if np.isnan(template).any():
			mask = np.isnan(template).any(axis=0)  # Channels that need to be run.
			self.compute_templates([unit_id], channel_ids[mask])
			template = self._templates[unit_id][:, channel_indices]

		template = template.copy()
		if return_scaled:
			gains = self.recording.get_channel_gains(channel_ids)
			offsets = self.recording.get_channel_offsets(channel_ids)
			template = template * gains[None, :] + offsets[None, :]

		return template

	def compute_templates(self, unit_ids=None, channel_ids=None) -> None:
		"""
		Computes the templates for the given units and channels.
		Doesn't return anything: updates TemplateExtractor._templates.

		@param unit_ids:
			The unit ids for which to compute the templates.
		@param channel_ids:
			The channel ids for which to compute the templates.
		"""

		recording = self.recording if channel_ids is None else self.recording.channel_slice(channel_ids)
		sorting = self.sorting if unit_ids is None else self.sorting.select_units(unit_ids)

		# TODO: max_spikes_per_unit
		# selected_spikes = si.waveform_extractor.select_random_spikes_uniformly(recording, sorting, self.params['max_spikes_per_unit'], self.nbefore, self.nafter)

		wvfs = si.extract_waveforms_to_buffers(recording, sorting.to_spike_vector(), sorting.unit_ids, self.nbefore, 1 + self.nafter, mode="memmap",
											   return_scaled=False, folder=self.folder, dtype=recording.dtype)

		for unit_id in sorting.unit_ids:
			unit_idx = self.sorting.id_to_index(unit_id)
			template = np.mean(wvfs[unit_id], axis=0)
			channel_indices = self.recording.ids_to_indices(recording.channel_ids)
			self._templates[unit_idx][:, channel_indices] = template
