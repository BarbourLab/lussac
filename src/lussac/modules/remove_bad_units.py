from typing import Any
from overrides import override
import numpy as np
from lussac.core import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si


class RemoveBadUnits(MonoSortingModule):
	"""
	Removes the bad units from the sorting object.
	The definition of "bad" unit is given by the parameters dictionary.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'wvf_extraction': {
				'ms_before': 1.0,
				'ms_after': 1.0,
				'max_spikes_per_unit': 500,
				'filter_band': [150, 9000]
			}
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		wvf_extraction_params = params.pop('wvf_extraction', {})
		if self.analyzer is None:
			self.precompute_analyzer(params)

		units_to_remove = np.zeros(self.sorting.get_num_units(), dtype=bool)
		reasons_for_removal = np.array([''] * self.sorting.get_num_units(), dtype=object)

		for attribute, p in params.items():
			if attribute == "all":
				units_to_remove[:] = True
				reasons_for_removal[:] += " ; all"
				break

			value = self.get_units_attribute_arr(attribute, p, **wvf_extraction_params)
			if 'min' in p:
				units_to_remove |= value < p['min']
				reasons_for_removal[value < p['min']] += f" ; {attribute} < {p['min']}"
			if 'max' in p:
				units_to_remove |= value > p['max']
				reasons_for_removal[value > p['max']] += f" ; {attribute} > {p['max']}"

		sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if not bad])
		bad_sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if bad])
		reasons_for_removal = ["Reason(s) for removal: " + reason[3:] for reason in reasons_for_removal[units_to_remove]]

		self._plot_bad_units(bad_sorting.unit_ids, reasons_for_removal)

		return sorting

	def precompute_analyzer(self, params: dict[str, Any]) -> None:
		params = self.update_params(params)
		wvf_extraction = params['wvf_extraction']

		self.create_analyzer(filter_band=wvf_extraction['filter_band'], cache_recording=True)
		self.analyzer.compute({
			'random_spikes': {'max_spikes_per_unit': wvf_extraction['max_spikes_per_unit']},
			'templates': {'ms_before': wvf_extraction['ms_before'], 'ms_after': wvf_extraction['ms_after']},
			'spike_amplitudes': {'peak_sign': 'both'}
		})

	def _plot_bad_units(self, bad_unit_ids, reasons_for_removal: list[str]) -> None:
		"""
		Plots the units that were removed.

		@param bad_unit_ids:
			The unit ids that were removed.
		"""
		if len(bad_unit_ids) == 0:
			return

		analyzer = self.analyzer.select_units(bad_unit_ids, format="memory")

		annotations = [{'text': reason, 'x': 0.6, 'y': 1.02, 'xref': "paper", 'yref': "paper", 'xanchor': "center", 'yanchor': "bottom", 'showarrow': False} for reason in reasons_for_removal]
		utils.plot_units(analyzer, filepath=f"{self.logs_folder}/bad_units", annotations_change=annotations)
