import numpy as np
import spikeinterface.core as si
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

			value = self.get_units_attribute_arr(attribute, p)
			if 'min' in p:
				units_to_remove |= value < p['min']
			if 'max' in p:
				units_to_remove |= value > p['max']

		sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if not bad])
		bad_sorting = self.sorting.select_units([unit_id for unit_id, bad in zip(self.sorting.unit_ids, units_to_remove) if bad])

		# TODO: Plot bad units.

		return sorting
