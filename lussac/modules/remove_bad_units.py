import numpy as np
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
from lussac.core.module import MonoSortingModule


class RemoveBadUnits(MonoSortingModule):
	"""
	Removes the bad units from the sorting object.
	The definition of "bad" unit is given by the parameters dictionary.
	"""

	def run(self, params: dict) -> si.BaseSorting:
		units_to_remove = np.zeros(self.sorting.get_num_units(), dtype=bool)

		for attribute, p in params.items():
			if attribute == "all":
				units_to_remove[:] = True
				break

			value = self._get_units_attribute(attribute, p)
			if 'min' in p:
				units_to_remove |= value < p['min']
			if 'max' in p:
				units_to_remove |= value > p['max']

		sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if not bad])
		bad_sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if bad])

		# TODO: Plot bad units.

		return sorting

	def _get_units_attribute(self, attribute: str, params: dict) -> np.ndarray:
		"""
		Gets the attribute for all the units.

		@param attribute: str
			The attribute name.
			- frequency (in Hz)
			- contamination (between 0 and 1)
			- amplitude (TODO)
			- amplitude_std (TODO)
		@return attribute: np.ndarray
			The attribute for all the units.
		"""
		recording = self.data.recording
		sorting = self.sorting
		if 'filter' in params:
			recording = spre.filter(recording, **params['filter'])

		wvf_extractor = self.extract_waveforms(sub_folder=attribute, **params['wvf_extraction']) if 'wvf_extraction' in params \
						else si.WaveformExtractor(recording, sorting)

		match attribute:
			case "frequency":  # Returns the firing rate of each unit (in Hz).
				n_spikes = np.array([len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids])
				frequencies = n_spikes * sorting.get_sampling_frequency() / recording.get_num_frames()
				return frequencies

			case "contamination":  # Returns the estimated contamination of each unit.
				censored_period, refractory_period = params['refractory_period']
				contamination = sqm.compute_refrac_period_violations(wvf_extractor, refractory_period, censored_period)[1]
				return np.array(list(contamination.values()))

			case "amplitude":  # Returns the amplitude of each unit on its best channel (TODO: unit).
				peak_sign = params['peak_sign'] if 'peak_sign' in params else "both"
				mode = params['mode'] if 'mode' in params else "extremum"
				amplitudes = spost.get_template_extremum_amplitude(wvf_extractor, peak_sign, mode)
				return np.array(list(amplitudes.values()))

			case "amplitude_std":
				peak_sign = params['peak_sign'] if 'peak_sign' in params else "both"
				chunk_duration = params['chunk_duration'] if 'chunk_duration' in params else '1s'
				n_jobs = params['n_jobs'] if 'n_jobs' in params else 6
				amplitudes = spost.compute_spike_amplitudes(wvf_extractor, peak_sign, outputs='by_unit', chunk_duration=chunk_duration, n_jobs=n_jobs)[0]
				std_amplitudes = [np.std(amplitudes[unit_id]) for unit_id in amplitudes.keys()]
				return np.array(std_amplitudes)

			case _:
				raise ValueError(f"Unknown attribute: {attribute}")
