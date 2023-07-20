from typing import Any
from overrides import override
import numpy as np
import numpy.typing as npt
from lussac.core import MonoSortingModule
import spikeinterface.core as si


class UnitsCategorization(MonoSortingModule):
	"""
	Categorizes units based on their properties.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		for category, rules in params.items():
			units_to_categorize = self._init_units_to_categorize()

			if category == "clear":
				self.sorting.set_property("lussac_category", None)
				continue

			for attribute, p in rules.items():
				value = self.get_units_attribute_arr(attribute, p)
				if 'min' in p:
					units_to_categorize &= value > p['min']
				if 'max' in p:
					units_to_categorize &= value < p['max']

			unit_ids = self.sorting.unit_ids[units_to_categorize]
			values = np.array([category]*len(unit_ids), dtype=object)  # dtype cannot be str, otherwise string length is restricted.
			self.sorting.set_property("lussac_category", values, ids=unit_ids, missing_value=None)

		return self.sorting

	def _init_units_to_categorize(self) -> npt.NDArray[np.integer]:
		"""
		Returns an array of True/False, where True means that the unit hasn't been categorized yet.

		@return units_to_categorize: np.ndarray[bool]
			Units that haven't been categorized yet.
		"""

		if self.sorting.get_property("lussac_category") is None:
			return np.ones(self.sorting.get_num_units(), dtype=bool)
		else:
			categories = self.sorting.get_property("lussac_category")
			return np.array([category is None for category in categories], dtype=bool)
