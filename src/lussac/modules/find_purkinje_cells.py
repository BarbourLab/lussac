from typing import Any
import numpy as np
from overrides import override
from lussac.core import MonoSortingModule
import spikeinterface.core as si
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm


class FindPurkinjeCells(MonoSortingModule):
	"""
	This module is meant purely for the cerebellar cortex.
	Purkinje cells are big GABAergic neurons that can fire both simple spikes and complex spikes.
	Since simple spikes and complex spikes have different shape, they will be clustered in different units.
	This module will find simple spikes and complex spikes coming from the same Purkinje cell,
	and label then into a 'lussac_purkinje' property in the sorting object.
	This is done by looking at the cross-correlogram between the two units: after a complex spike,
	there is always a pause of at least 8-10 ms before simple spike activity resumes.
	Also, when high-pass filtered aggressively, the templates are nearly identical.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'cross_corr_pause': [0.0, 8.0],
			'threshold': 0.4,
			'ss_min_fr': 40.0,
			'cs_min_fr': 0.5,
			'cs_max_fr': 3.0
		}

	@override
	def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
		params = super().update_params(params)
		if params['ss_min_fr'] <= params['cs_max_fr']:
			raise ValueError("Error in 'find_purkinje_cells': 'ss_min_fr' must be greater than 'cs_max_fr'")
		if params['threshold'] < 0.0:
			raise ValueError("Error in 'find_purkinje_cells': 'threshold' cannot be negative")

		return params

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		wvf_extractor = si.WaveformExtractor(self.recording, self.sorting, allow_unfiltered=True)
		firing_rates = sqm.compute_firing_rates(wvf_extractor)

		putative_ss_units = [unit_id for unit_id, mean_fr in firing_rates.items() if mean_fr >= params['ss_min_fr']]
		putative_cs_units = [unit_id for unit_id, mean_fr in firing_rates.items() if params['cs_min_fr'] <= mean_fr <= params['cs_max_fr']]

		sorting = self.sorting.select_units([*putative_ss_units, *putative_cs_units])
		correlograms, bins = spost.compute_correlograms(sorting, window_ms=25.0, bin_ms=1.0, method="numba")
		mask = ((bins >= params['cross_corr_pause'][0]) & (bins < params['cross_corr_pause'][1]))[:-1]

		ss_cs_pairs = []
		for ss_id in putative_ss_units:
			for cs_id in putative_cs_units:
				ss_ind = sorting.id_to_index(ss_id)
				cs_ind = sorting.id_to_index(cs_id)
				cross_corr = correlograms[ss_ind, cs_ind, :]

				baseline = np.median(cross_corr[bins[:-1] < 0.0])
				if np.median(cross_corr[mask]) < baseline * params['threshold']:  # Check for pause.
					if np.median(cross_corr[mask[::-1]] < baseline * params['threshold']):  # Check for asymmetry.
						continue
					ss_cs_pairs.append((ss_id, cs_id))  # TODO: Also check templates?

		lussac_purkinje = np.empty(self.sorting.get_num_units(), dtype=object)
		paired_units = np.unique(np.array(ss_cs_pairs).flatten())
		for unit_id in paired_units:
			unit_ind = self.sorting.id_to_index(unit_id)
			mask = np.array([unit_id in ss_cs_pairs[i] for i in range(len(ss_cs_pairs))])
			text = " ; ".join([f"{ss_id}-{cs_id}" for ss_id, cs_id in np.array(ss_cs_pairs)[mask]])
			lussac_purkinje[unit_ind] = text

		self.sorting.set_property('lussac_purkinje', lussac_purkinje)
		return self.sorting
