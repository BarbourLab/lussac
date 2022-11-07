import copy
import os
from lussac.core.lussac_data import MonoSortingData
from lussac.modules.remove_redundant_units import RemoveRedundantUnits


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = RemoveRedundantUnits("test_rru_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_remove_redundant_units(mono_sorting_data: MonoSortingData) -> None:
	data = copy.deepcopy(mono_sorting_data)
	data.sorting = data.sorting.select_units([80, 81, 82, 83, 84, 85])
	params = {'wvf_extraction': None, 'arguments': {'align': False, 'agreement_threshold': 0.1, 'duplicate_threshold': 0.7, 'remove_strategy': 'max_spikes'}}

	module = RemoveRedundantUnits("test_rru", data, "all")
	assert not os.path.exists(f"{module.logs_folder}/redundant_units.html")
	sorting = module.run(params)
	assert os.path.exists(f"{module.logs_folder}/redundant_units.html")

	assert sorting.get_num_units() < data.sorting.get_num_units()
