import numpy as np


class WaveformExtractor:

	def __init__(self, data):
		"""
		Creates a new WaveformExtractor instance.

		@param data (PhyData):
			The data object.
		"""

		self.data = data
		self.mean_wvfs = dict()
		self.used_params = dict()
		self.default_params = {
			'ms_before': 2,
			'ms_after': 2,
			'max_spikes_per_unit': None,
			'max_channels_per_waveforms': None,
			'return_idx': False
		}


	def _save(self, wvf, params: dict, sorting: int, unit_id: int):
		"""
		TODO
		"""

		if sorting not in self.mean_wvfs:
			self.mean_wvfs[sorting] = dict()
			self.used_params[sorting] = dict()

		self.mean_wvfs[sorting][unit_id] = wvf
		self.used_params[sorting][unit_id] = params


	def clear(self):
		"""
		Clears all saved waveforms.
		"""

		del self.mean_wvfs
		self.mean_wvfs = dict()


	def get_units_waveforms(self, unit_ids: list, sorting: int, **params):
		"""
		Runs WaveformExtractor.get_unit_waveforms for all given units.

		@return waveforms (list of np.ndarray) [n_units][n_waveforms, n_channels, time]:
			The extracted waveforms from the units.

		For more details, see WaveformExtractor.get_unit_waveforms().
		"""

		return [self.get_unit_waveforms(unit_id, sorting, **params) for unit_id in unit_ids]


	def get_units_mean_waveform(self, unit_ids: list, sorting: int, save_mean: bool=True, **params):
		"""
		Runs WaveformExtractor.get_unit_mean_waveform for all given units.

		@param save_mean (bool):
			If True, will save the computed mean to WaveformExtractor.mean_wvfs
		@return waveforms (np.ndarray) [n_units, n_channels, time]:
			The extracted mean waveforms from the units.

		For more details, see WaveformExtractor.get_unit_waveforms().
		"""

		return np.array([self.get_unit_mean_waveform(unit_id, sorting, save_mean, **params) for unit_id in unit_ids], dtype=np.float32)


	def get_unit_mean_waveform(self, unit_id: int, sorting: int, save_mean: bool=True, use_saved: bool=True, **params):
		"""
		Returns the mean waveform of a unit.

		@param save_mean (bool):
			If True, will save the computed mean to WaveformExtractor.mean_wvfs
		@param use_saved (bool):
			If True and a saved mean waveform exists, will return it instead of recomputing it.
		@return mean_waveform (np.ndarray) [n_channels, time]:
			The computed mean from extracted waveforms from the unit.

		For more details, see WaveformExtractor.get_unit_waveforms().
		"""

		params = {**self.default_params, **params}

		if use_saved and sorting in self.mean_wvfs and unit_id in self.mean_wvfs[sorting]:
			return self.get_stored_mean_waveform(unit_id, sorting, **params)

		mean_wvf = np.mean(self.get_unit_waveforms(unit_id, sorting, **params), axis=0, dtype=np.float32)
		if save_mean:
			self._save(mean_wvf, params, sorting, unit_id)

		return mean_wvf


	def get_stored_mean_waveform(self, unit_id: int, sorting: int, **params):
		"""

		"""

		if sorting not in self.mean_wvfs or unit_id not in self.mean_wvfs[sorting]:
			return None

		used_params = self.used_params[sorting][unit_id]

		if self._compare_params(used_params, params):

			new_params = {
				'ms_before': max(used_params['ms_before'], params['ms_before']),
				'ms_after':  max(used_params['ms_after'], params['ms_after']),
				'max_spikes_per_unit': max(used_params['max_spikes_per_unit'], params['max_spikes_per_unit']),
				'max_channels_per_waveforms': None if (used_params['max_channels_per_waveforms'] == None or params['max_channels_per_waveforms'] == None)
													else max(used_params['max_channels_per_waveforms'], params['max_channels_per_waveforms'])
			}

			mean_wvf = self.get_unit_mean_waveform(unit_id, sorting, save_mean=True, use_saved=False, **new_params)
			used_params = new_params

		start = int((used_params['ms_before'] - params['ms_before']) * self.data.sampling_f * 1e-3)
		end = int((used_params['ms_before'] + params['ms_after']) * self.data.sampling_f * 1e-3) + 1

		return self.mean_wvfs[sorting][unit_id][:params['max_channels_per_waveforms'], start:end]


	def _compare_params(self, params1: dict, params2: dict):
		"""

		"""

		if params1['ms_before'] < params2['ms_before']:
			return True
		if params1['ms_after'] < params2['ms_after']:
			return True
		if params1['max_spikes_per_unit'] < params2['max_spikes_per_unit']:
			return True
		if params1['max_channels_per_waveforms'] != None and params2['max_channels_per_waveforms'] == None:
			return True
		if params1['max_channels_per_waveforms'] != None and params2['max_channels_per_waveforms'] > params1['max_channels_per_waveforms']:
			return True

		return False


	def get_unit_waveforms(self, unit_id: int, sorting: int, **params):
		"""
		Returns the waveforms of a unit.

		@param unit_id (int):
			ID of unit to extract the waveforms from.
		@param sorting (int):
			Sorting the unit comes from.
		@param ms_before (float):
			How much time before the spike center to extract (in ms).
		@param ms_after (float):
			How much time after the spike center to extract (in ms).
		@param max_spikes_per_unit (int or None):
			If int, will extract a random subsample of this many spikes.
			If None, will extract all the spikes.
		@param max_channels_per_waveforms (int or list or None):
			If int, will extract from this many channels (will take the best ones).
			If list (of ints), will extract from the given channels.
			If None, will extract from all channels.
		@param return_idx (bool):
			If true, will return some useful info about indices as well as the waveforms.
			If false, will  only return the waveforms.

		@return waveforms (np.ndarray) [n_waveforms, n_channels, time]:
			The extracted waveforms from the unit (dtype taken from recording).
		@return channels_idx (np.ndarray) [n_channels]: (ONLY if return_idx==True)
			The channels indices used for extraction.
			Will be ordered by best channels first if max_channels_per_waveforms!= None and less than maximum number of channels.
		"""

		params = {**self.default_params, **params}

		spike_train = self.data._sortings[sorting].get_unit_spike_train(unit_id)
		if params['max_spikes_per_unit'] != None and params['max_spikes_per_unit'] < len(spike_train):
			spike_train = np.sort(np.random.choice(spike_train, params['max_spikes_per_unit'], False))

		return self.get_waveforms_from_spiketrain(spike_train, **params)


	def get_waveforms_from_spiketrain(self, spike_train: np.ndarray, **params):
		"""
		Returns the waveforms at certain time points (from spike train).

		@param spike_train (np.ndarray) [n_spikes]:
			Timings of spikes (in sampling time).
		@param ms_before (float):
			How much time before the spike center to extract (in ms).
		@param ms_after (float):
			How much time after the spike center to extract (in ms).
		@param max_channels_per_waveforms (int or list or None):
			If int, will extract from this many channels (will take the best ones). WARNING: if spike train isn't big, best channel might fail.
			If list (of ints), will extract from the given channels.
			If None, will extract from all channels.
		@param return_idx (bool):
			If true, will return some useful info about indices as well as the waveforms.
			If false, will  only return the waveforms.

		@return waveforms (np.ndarray) [n_waveforms, n_channels, time]:
			The extracted waveforms from the unit (dtype taken from recording).
		@return channels_idx (np.ndarray) [n_channels]: (ONLY if return_idx==True)
			The channels indices used for extraction.
			Will be ordered by best channels first if max_channels_per_waveforms!= None and less than maximum number of channels.
		"""

		params = {**self.default_params, **params}

		recording = self.data.recording
		before	= int(round(params['ms_before'] * recording.get_sampling_frequency() / 1e3))
		after	= int(round(params['ms_after'] * recording.get_sampling_frequency() / 1e3)) + 1

		t_max	= recording.get_num_frames()
		indices = np.tile(spike_train[:, None], (1, before+after)) + np.arange(-before, after)
		indices[:before] = np.where(indices[:before]<0, 0, indices[:before])			# Check that first waveforms don't have negative indices.
		indices[-after:] = np.where(indices[-after:]>=t_max, t_max-1, indices[-after:])	# Check that last waveforms don't go beyond recording.

		waveforms = np.swapaxes(recording.get_traces()[:, indices.astype(np.uint64)], 0, 1)

		if isinstance(params['max_channels_per_waveforms'], list) or isinstance(params['max_channels_per_waveforms'], np.ndarray):
			channels_idx = np.array(params['max_channels_per_waveforms'], dtype=np.uint16)
		elif isinstance(params['max_channels_per_waveforms'], int) and params['max_channels_per_waveforms'] < waveforms.shape[1]:
			mean = np.abs(np.mean(waveforms, axis=0))
			max_amplitude = np.max(mean, axis=1)
			channels_idx = np.argsort(max_amplitude)[:-params['max_channels_per_waveforms']-1:-1]
		else:
			channels_idx = np.arange(waveforms.shape[1])

		waveforms = waveforms[:, channels_idx]

		if params['return_idx']:
			return waveforms, channels_idx
		else:
			return waveforms

