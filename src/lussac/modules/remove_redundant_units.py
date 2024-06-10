from typing import Any
import numpy as np
from overrides import override
from lussac.core import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur


class RemoveRedundantUnits(MonoSortingModule):
	"""
	Removes units that are redundant with other units in the same sorting
	(i.e. they share similar spike timings over a certain threshold).
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'wvf_extraction': {
				'ms_before': 1.0,
				'ms_after': 1.5,
				'max_spikes_per_unit': 500,
				'filter_band': None
			},
			'arguments': {
				'align': True,
				'delta_time': 0.3,
				'agreement_threshold': 0.1,
				'duplicate_threshold': 0.7,
				'remove_strategy': 'highest_amplitude'
			}
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		if self.analyzer is None:
			self.precompute_analyzer(params)

		new_sorting, redundant_unit_pairs = scur.remove_redundant_units(self.analyzer, extra_outputs=True, **params['arguments'])

		redundant_unit_ids = [unit_id for unit_id in self.sorting.unit_ids if unit_id not in new_sorting.unit_ids]
		redundant_sorting = self.sorting.select_units(redundant_unit_ids)
		redundancies = self._get_redundancies(redundant_unit_ids, redundant_unit_pairs)
		self._plot_redundant_units(redundant_sorting, redundancies)

		return self.sorting.select_units(new_sorting.unit_ids)  # can't use `new_sorting` because parent is SharedMemorySorting, which can't be pickled

	def precompute_analyzer(self, params: dict[str, Any]) -> None:
		params = self.update_params(params)
		wvf_extraction = params['wvf_extraction']

		self.create_analyzer(filter_band=wvf_extraction['filter_band'] if wvf_extraction is not None else None)
		if wvf_extraction is not None:
			self.analyzer.compute({
				'random_spikes': {'max_spikes_per_unit': params['wvf_extraction']['max_spikes_per_unit']},
				'templates': {'ms_before': params['wvf_extraction']['ms_before'], 'ms_after': params['wvf_extraction']['ms_after']}
			})

	@staticmethod
	def _get_redundancies(redundant_unit_ids: list, redundant_unit_pairs: list[list]) -> dict:
		"""
		For each redundant units, finds which units it is redundant with.

		@param redundant_unit_ids: list
			A list of all redundant units' id.
		@param redundant_unit_pairs: list[list]
			A list of pairs of redundant units.
		@return redundancies: dict
			For each redundant unit (key = id), a list of the units it is redundant with (value = list of ids).
		"""

		redundancies = {}
		for unit_id in redundant_unit_ids:
			redundant_with = []
			for pair in redundant_unit_pairs:
				if unit_id in pair:
					redundant_with.append(pair[0] if unit_id == pair[1] else pair[1])
			redundancies[unit_id] = redundant_with

		return redundancies

	def _plot_redundant_units(self, redundant_sorting: si.BaseSorting, redundancies: dict) -> None:
		"""
		Plots the units that were removed.

		@param redundant_sorting: si.BaseSorting
			The sorting object containing the redundant units.
		"""
		if redundant_sorting.get_num_units() == 0:
			return

		analyzer = self.analyzer.select_units(redundant_sorting.unit_ids, format="memory")
		if not analyzer.has_extension("fast_templates"):
			analyzer.compute({
				'random_spikes': {'max_spikes_per_unit': 500},
				'templates': {'ms_before': 1.5, 'ms_after': 2.5},
			})

		annotations = [{'text': f"Unit {unit_id} is redundant with unit(s): {' '.join(np.array(redundancies[unit_id]).astype(str))}", 'x': 0.6, 'y': 1.07,
						'xref': "paper", 'yref': "paper", 'showarrow': False} for unit_id in redundant_sorting.unit_ids]
		utils.plot_units(analyzer, filepath=f"{self.logs_folder}/redundant_units", annotations_change=annotations)
