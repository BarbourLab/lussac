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

		for category, rules in params.items():
			units_to_categorize = self._init_units_to_categorize()

			if category == "clear":
				self.sorting.delete_property("lussac_category")
				continue

			for attribute, p in rules.items():
				value = self.get_units_attribute_arr(attribute, p, **wvf_extraction_params)
				if 'min' in p:
					units_to_categorize &= value > p['min']
				if 'max' in p:
					units_to_categorize &= value < p['max']

			unit_ids = self.sorting.unit_ids[units_to_categorize]
			values = np.array([category]*len(unit_ids), dtype=object)  # dtype cannot be str, otherwise string length is restricted.
			self.sorting.set_property("lussac_category", values, ids=unit_ids, missing_value=None)

		return self.sorting

	def precompute_analyzer(self, params: dict[str, Any]) -> None:
		params = self.update_params(params)
		wvf_extraction = params['wvf_extraction']

		self.create_analyzer(filter_band=wvf_extraction['filter_band'])

		attributes = list(params.keys())
		if "amplitude" in attributes or "SNR" in attributes or "sd_ratio" in attributes:
			self.analyzer.compute({
				'random_spikes': {'max_spikes_per_unit': wvf_extraction['max_spikes_per_unit']},
				'templates': {'ms_before': wvf_extraction['ms_before'], 'ms_after': wvf_extraction['ms_after']}

			})
		if "sd_ratio" in attributes:
			self.analyzer.compute("spike_amplitudes", peak_sign="both")

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
