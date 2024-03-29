import os
from lussac.core import MonoSortingData
from lussac.modules import RemoveRedundantUnits


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = RemoveRedundantUnits("test_rru_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_remove_redundant_units(mono_sorting_data: MonoSortingData) -> None:
	params = {'wvf_extraction': None, 'arguments': {'align': False, 'agreement_threshold': 0.1, 'duplicate_threshold': 0.7, 'remove_strategy': 'max_spikes'}}

	# Create a smaller data object for testing (faster).
	data = MonoSortingData(mono_sorting_data.data, mono_sorting_data.sorting.select_units([80, 81, 84, 85]).frame_slice(0, 3_000_000))
	module = RemoveRedundantUnits("test_rru", data, "all")

	assert not os.path.exists(f"{module.logs_folder}/redundant_units.html")
	sorting = module.run(params)
	assert os.path.exists(f"{module.logs_folder}/redundant_units.html")

	assert sorting.get_num_units() < data.sorting.get_num_units()


def test_get_redundancies() -> None:
	assert RemoveRedundantUnits._get_redundancies([], []) == {}
	assert RemoveRedundantUnits._get_redundancies([1], []) == {1: []}
	assert RemoveRedundantUnits._get_redundancies([1, 2], [[1, 5], [2, 8], [8, 13], [2, 7], [3, 4]]) == {1: [5], 2: [8, 7]}
