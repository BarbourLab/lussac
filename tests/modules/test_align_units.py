import copy
import os
from lussac.core.lussac_data import MonoSortingData
from lussac.modules.align_units import AlignUnits


params = {
	'wvf_extraction': {
		'ms_before': 1.5,
		'ms_after': 2.5,
		'max_spikes_per_unit': 20
	},
	'filter': [150, 6000],
	'threshold': 0.5
}


def test_remove_bad_units(mono_sorting_data: MonoSortingData) -> None:
	data = copy.deepcopy(mono_sorting_data)
	data.sorting = data.sorting.select_units([14, 22, 70, 71])

	module = AlignUnits("test_align_units", data, "all")
	assert not os.path.exists(f"{module.logs_folder}/alignment.html")
	sorting = module.run(params)
	assert os.path.exists(f"{module.logs_folder}/alignment.html")

	assert sorting.get_num_units() == data.sorting.get_num_units()
