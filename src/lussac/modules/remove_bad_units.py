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
		return {}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		units_to_remove = np.zeros(self.sorting.get_num_units(), dtype=bool)
		reasons_for_removal = np.array([''] * self.sorting.get_num_units(), dtype=object)

		for attribute, p in params.items():
			if attribute == "all":
				units_to_remove[:] = True
				reasons_for_removal[:] += " ; all"
				break

			value = self.get_units_attribute_arr(attribute, p)
			if 'min' in p:
				units_to_remove |= value < p['min']
				reasons_for_removal[value < p['min']] += f" ; {attribute} < {p['min']}"
			if 'max' in p:
				units_to_remove |= value > p['max']
				reasons_for_removal[value > p['max']] += f" ; {attribute} > {p['max']}"

		sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if not bad])
		bad_sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if bad])
		reasons_for_removal = ["Reason(s) for removal: " + reason[3:] for reason in reasons_for_removal[units_to_remove]]

		self._plot_bad_units(bad_sorting, reasons_for_removal)

		return sorting

	def _plot_bad_units(self, bad_sorting: si.BaseSorting, reasons_for_removal: list[str]) -> None:
		"""
		Plots the units that were removed.

		@param bad_sorting: si.BaseSorting
			The sorting object containing the bad units.
		"""

		wvf_extractor = self.extract_waveforms(sorting=bad_sorting, ms_before=1.5, ms_after=2.5, max_spikes_per_unit=500, sparse=False)
		annotations = [{'text': reason, 'x': 0.6, 'y': 1.02, 'xref': "paper", 'yref': "paper", 'xanchor': "center", 'yanchor': "bottom", 'showarrow': False} for reason in reasons_for_removal]
		utils.plot_units(wvf_extractor, filepath=f"{self.logs_folder}/bad_units", annotations_change=annotations)
