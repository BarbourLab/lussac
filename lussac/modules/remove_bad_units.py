from typing import Any
import numpy as np
from lussac.core.module import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si


class RemoveBadUnits(MonoSortingModule):
	"""
	Removes the bad units from the sorting object.
	The definition of "bad" unit is given by the parameters dictionary.
	"""

	@property
	def default_params(self) -> dict[str, Any]:
		return {}

	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		units_to_remove = np.zeros(self.sorting.get_num_units(), dtype=bool)

		for attribute, p in params.items():
			if attribute == "all":
				units_to_remove[:] = True
				break

			value = self.get_units_attribute_arr(attribute, p)
			if 'min' in p:
				units_to_remove |= value < p['min']
			if 'max' in p:
				units_to_remove |= value > p['max']

		sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if not bad])
		bad_sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if bad])

		self._plot_bad_units(bad_sorting)

		return sorting

	def _plot_bad_units(self, bad_sorting: si.BaseSorting) -> None:
		"""
		Plots the units that were removed.

		@param bad_sorting: si.BaseSorting
			The sorting object containing the bad units.
		"""

		if bad_sorting.get_num_units() == 0:
			return  # TODO: Should be removed with SpikeInterface PR #1055.

		wvf_extractor = self.extract_waveforms(sorting=bad_sorting, ms_before=1.5, ms_after=2.5, max_spikes_per_unit=500)
		utils.plot_units(wvf_extractor, filepath=f"{self.logs_folder}/bad_units")
