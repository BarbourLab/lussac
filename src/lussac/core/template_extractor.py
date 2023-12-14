import math
import pathlib
import tempfile
from typing import Any, Sequence
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
import spikeinterface.core as si


class TemplateExtractor:
	"""
	Lazy template extractor for fast computation.
	Only works for average

	Attributes
		recording		The recording extractor containing the voltage traces.
		sorting			The sorting extractor containing the spike times.
		params			The waveforms parameters for extraction.
		_templates		The templates memory map.
		_best_channels	The best channels for each unit.
	"""

	__slots__ = "recording", "sorting", "_folder", "params", "_templates", "_best_channels"
	recording: si.BaseRecording
	sorting: si.BaseSorting
	_folder: tempfile.TemporaryDirectory
	params: dict[str, Any]
	_templates: np.memmap | np.ndarray
	_best_channels: np.ndarray

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
		folder.mkdir(parents=True, exist_ok=True)
		self._folder = tempfile.TemporaryDirectory(dir=folder)
		if params is None:
			params = {}
		self.set_params(**params, templates_dtype=templates_dtype)

	def __del__(self) -> None:
		"""
		Deletes the temporary folder.
		"""

		del self._templates
		self._folder.cleanup()

	@property
	def folder(self) -> pathlib.Path:
		"""
		Returns the path to the folder where the templates are saved.

		@return: Path
			The path to the folder where the templates are saved.
		"""

		return pathlib.Path(self._folder.name)

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

	def _setup_templates(self, dtype: npt.DTypeLike = np.float32) -> None:
		"""
		Sets up the templates memory map (creates directory, creates memmap with dtype and shape, sets values to nan).

		@param dtype: DTypeLike
			The dtype of the templates (by default: np.float32).
		"""
		if hasattr(self, "_templates"):
			del self._templates  # Prevents crash in Windows where cannot open "mode='w+'" if file already opened.

		self.folder.mkdir(parents=True, exist_ok=True)

		if self.num_units * self.num_channels == 0:
			self._templates = np.empty((self.num_units, self.nsamples, self.num_channels), dtype=dtype)
		else:
			self._templates = np.memmap(str(self.folder / "templates.npy"), dtype=dtype, mode='w+', shape=(self.num_units, self.nsamples, self.num_channels))

		self._templates[:] = np.nan
		self._best_channels = np.zeros((self.num_units, self.num_channels), dtype=self.recording.channel_ids.dtype)

	def set_params(self, ms_before: float = 1.0, ms_after: float = 2.0, max_spikes_per_unit: int | None = 1_000, max_spikes_sparsity: int = 100,
				   templates_dtype: npt.DTypeLike | None = None) -> None:
		"""
		Sets the parameters for the waveforms extraction.
		Will reset the already-computed templates.

		@param ms_before: float
			Number of ms to retrieve before the spike events.
		@param ms_after: float
			Number of ms to retrieve after the spike events.
		@param max_spikes_per_unit: int
			Maximum number of spikes to retrieve per unit.
		@param max_spikes_sparsity: int
			Maximum number of spikes to retrieve to compute maximum amplitude per channel.
		"""

		self.params = {
			'ms_before': ms_before,
			'ms_after': ms_after,
			'max_spikes_per_unit': max_spikes_per_unit,
			'max_spikes_sparsity': max_spikes_sparsity
		}

		# Reset the templates with new params.
		if templates_dtype is None:
			templates_dtype = self._templates.dtype if hasattr(self, '_templates') else np.float32
		self._setup_templates(templates_dtype)

	def get_template(self, unit_id, channel_ids: Sequence | None = None, return_scaled: bool = False) -> np.ndarray:
		"""
		Returns the template for a given unit and channels.
		If not computed, will compute it on the fly.
		Returns a copy array.

		@param unit_id:
			The unit id for which to return the template.
		@param channel_ids:
			The channel ids for which to return the template.
			If None, will return the template for all channels (default: None).
		@param return_scaled: bool
			If True, will return the templates scaled to µV (default: False).
		@return template: array (n_samples_time, n_channels)
			The template for the given unit and channels (as a copy).
		"""

		return self.get_templates([unit_id], channel_ids, return_scaled)[0]

	def get_templates(self, unit_ids: Sequence | None = None, channel_ids: Sequence | None = None, return_scaled: bool = False) -> np.ndarray:
		"""
		Returns the templates for the given units and channels.
		If not computed, will compute them on the fly.
		Returns a copy array.

		@param unit_ids: Sequence | None
			The unit ids for which to return the templates.
			If None, will return the templates for all units (default: None).
		@param channel_ids:
			The channel ids for which to return the templates.
			If None, will return the templates for all channels (default: None).
		@param return_scaled: bool
			If True, will return the templates scaled to µV (default: False).
		@return templates: array (n_units, n_samples_time, n_channels)
			The templates for the given units and channels (as a copy).
		"""

		if unit_ids is None:
			unit_ids = self.unit_ids
		if channel_ids is None:
			channel_ids = self.channel_ids
		unit_ids = np.array(unit_ids)
		channel_ids = np.array(channel_ids)

		unit_indices = self.sorting.ids_to_indices(unit_ids)
		channel_indices = self.recording.ids_to_indices(channel_ids)
		templates = self._templates[unit_indices][:, :, channel_indices]

		if np.isnan(templates).any():
			self.compute_templates(unit_ids, channel_ids)
			templates = self._templates[unit_indices][:, :, channel_indices]

		templates = templates.copy()
		if return_scaled:
			gains = self.recording.get_channel_gains(channel_ids)
			offsets = self.recording.get_channel_offsets(channel_ids)
			templates = templates * gains[None, None, :] + offsets[None, None, :]

		return templates

	def compute_templates(self, unit_ids: Sequence | None = None, channel_ids: Sequence | None = None) -> None:
		"""
		Computes the templates for the given units and channels.
		Doesn't return anything: updates TemplateExtractor._templates.

		@param unit_ids: Sequence | None
			The unit ids for which to compute the templates.
			If None, will compute the templates for all units (default: None).
		@param channel_ids:
			The channel ids for which to compute the templates.
			If None, will compute the templates for all channels (default: None).
		"""

		recording = self.recording if channel_ids is None else self.recording.channel_slice(channel_ids)
		sorting = self.sorting if unit_ids is None else self.sorting.select_units(unit_ids)
		channel_indices = self.recording.ids_to_indices(recording.channel_ids)
		unit_indices = self.sorting.ids_to_indices(sorting.unit_ids)

		if (~np.isnan(self._templates[unit_indices])).all(axis=(1, 2)).any():  # Some units have already been computed
			mask_units = np.isnan(self._templates[unit_indices]).any(axis=(1, 2))
			self.compute_templates(sorting.unit_ids[mask_units], channel_ids)
			return

		if (~np.isnan(self._templates[:, :, channel_indices])).all(axis=(0, 1)).any():  # Some channels have already been computed
			mask_channels = np.isnan(self._templates[:, :, channel_indices]).any(axis=(0, 1))
			self.compute_templates(unit_ids, recording.channel_ids[mask_channels])
			return

		if self.params['max_spikes_per_unit'] is None:
			spike_vector = sorting.to_spike_vector()
		else:
			selected_spikes = {}
			for unit_id in sorting.unit_ids:
				spike_train = sorting.get_unit_spike_train(unit_id)
				n_spikes = min(len(spike_train), self.params['max_spikes_per_unit'])
				selected_spikes[unit_id] = np.sort(np.random.choice(spike_train, n_spikes, replace=False))
			spike_vector = si.NumpySorting.from_unit_dict(selected_spikes, self.sampling_frequency).to_spike_vector()

		(self.folder / "waveforms").mkdir(exist_ok=True, parents=True)
		wvfs = si.extract_waveforms_to_buffers(recording, spike_vector, sorting.unit_ids, self.nbefore, 1 + self.nafter, mode="memmap",
											   return_scaled=False, folder=self.folder / "waveforms", dtype=recording.dtype, n_jobs=1)

		for unit_idx, unit_id in zip(unit_indices, sorting.unit_ids):
			template = np.mean(wvfs[unit_id], axis=0)
			self._templates[unit_idx][:, channel_indices] = template

	def get_unit_best_channels(self, unit_id, **kwargs) -> np.ndarray:
		"""
		Returns the best channel for the given unit.
		If not computed, will compute it on the fly.
		Returns a copy array.

		@param unit_id: int
			The unit id for which to return the best channels.
		@param kwargs:
			Keyword arguments to be passed to compute_best_channels.
		@return best_channel: array (n_channels,)
			The best channel for the given unit (as a copy).
		"""

		return self.get_units_best_channels([unit_id], **kwargs)[0]

	def get_units_best_channels(self, unit_ids: Sequence | None = None, **kwargs) -> np.ndarray:
		"""
		Returns the best channels for the given units.
		If not computed, will compute them on the fly.
		Returns a copy array.

		@param unit_ids: Sequence | None
			The unit ids for which to return the best channels.
			If None, will return the best channels for all units (default: None).
		@param kwargs:
			Keyword arguments to be passed to compute_best_channels.
		@return best_channels: array (n_units, n_channels)
			The best channels for the given units (as a copy).
		"""

		if unit_ids is None:
			unit_ids = self.unit_ids
		unit_ids = np.array(unit_ids)

		unit_indices = self.sorting.ids_to_indices(unit_ids)
		best_channels = self._best_channels[unit_indices]

		if np.all(best_channels == best_channels[:, 0, None], axis=1).any():
			self.compute_best_channels(unit_ids, **kwargs)
			best_channels = self._best_channels[unit_indices]

		return best_channels.copy()

	def compute_best_channels(self, unit_ids: Sequence | None = None, highpass_filter: float = 150.,
							 ms_before: float | None = None, ms_after: float | None = None) -> None:
		"""
		Computes the best channels (by looking at the peak amplitude after a high-pass filter) for the given units.
		Extracts a very small number of waveforms, average them to make templates, then compute the peak amplitude on each channel.
		Channels are ordered by decreasing peak amplitude.
		Doesn't return anything: updates TemplateExtractor._best_channels.

		@param unit_ids: Sequence | None
			The unit ids for which to compute the best channels.
			If None, will compute the best channels for all units (default: None).
		@param highpass_filter: float
			Cutoff frequency for the high-pass filter (in Hz).
		@param ms_before: float | None
			Number of ms to include before the spike events (default: None).
			If None, will use the values from the parameters.
		@param ms_after: float | None
			Number of ms to include after the spike events (default: None).
			If None, will use the values from the parameters.
		"""

		if unit_ids is None:
			unit_ids = self.unit_ids
		unit_ids = np.array(unit_ids)

		if len(unit_ids) == 0:
			return

		unit_indices = self.sorting.ids_to_indices(unit_ids)
		if not np.any(mask := np.all(self._best_channels[unit_indices] == self._best_channels[unit_indices, 0, None], axis=1)):
			# Some units have already been computed
			self.compute_best_channels(unit_ids[mask], highpass_filter)
			return

		sorting = self.sorting.select_units(unit_ids)

		params = {
			'ms_before': self.params['ms_before'] if ms_before is None else ms_before,
			'ms_after': self.params['ms_after'] if ms_after is None else ms_after,
			'max_spikes_per_unit': self.params['max_spikes_sparsity'],
			'precompute_template': ("average", ),
			'return_scaled': False,
			'sparse': False,
			'allow_unfiltered': True
		}
		wvf_extractor = si.extract_waveforms(self.recording, sorting, mode="memory", n_jobs=1, **params)

		templates = wvf_extractor.get_all_templates(mode="average")
		templates = (templates - gaussian_filter1d(templates, sigma=self.sampling_frequency / (2 * np.pi * highpass_filter), axis=1))
		best_channels_indices = np.argsort(np.max(np.abs(templates), axis=1), axis=1)[:, ::-1]
		best_channels = self.recording.channel_ids[best_channels_indices]

		self._best_channels[unit_indices] = best_channels
