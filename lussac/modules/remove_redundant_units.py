from lussac.core.module import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur


class RemoveRedundantUnits(MonoSortingModule):
	"""
	Removes units that are redundant with other units in the same sorting
	(i.e. they share similar spike timings over a certain threshold).
	"""

	def run(self, params: dict) -> si.BaseSorting:
		sorting_or_wvf_extractor = self.extract_waveforms(**params['wvf_extraction']) if 'wvf_extraction' in params else self.sorting

		new_sorting = scur.remove_redundant_units(sorting_or_wvf_extractor, **params['arguments'])

		redundant_unit_ids = [unit_id for unit_id in self.sorting.unit_ids if unit_id not in new_sorting.unit_ids]
		redundant_sorting = self.sorting.select_units(redundant_unit_ids)
		self._plot_redundant_units(redundant_sorting)

		return new_sorting

	def _plot_redundant_units(self, redundant_sorting: si.BaseSorting) -> None:
		"""
		Plots the units that were removed.

		@param redundant_sorting: si.BaseSorting
			The sorting object containing the redundant units.
		"""

		if redundant_sorting.get_num_units() == 0:
			return

		wvf_extractor = self.extract_waveforms(sorting=redundant_sorting, ms_before=1.5, ms_after=2.5, max_spikes_per_unit=500)
		utils.plot_units(wvf_extractor, filepath=f"{self.logs_folder}/redundant_units")  # TODO: Add annotation to say with which unit it is redundant.
