import os
from lussac.core import MonoSortingData
from lussac.modules import RemoveBadUnits


PARAMS = {
	'wvf_extraction': {
		'ms_before': 1.0,
		'ms_after': 1.0,
		'max_spikes_per_unit': 50,
		'filter': None
	},
	"firing_rate": {
		"min": 0.5
	},
	"contamination": {
		"refractory_period": [0.5, 1.0],
		"max": 0.25
	},
	"amplitude": {
		"min": 20
	},
	"SNR": {
		"min": 1.2
	},
	"sd_ratio": {
		"max": 2.0
	}
}


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = RemoveBadUnits("test_rbu_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_remove_bad_units(mono_sorting_data: MonoSortingData) -> None:
	# Create a smaller data object for testing (faster).
	data = mono_sorting_data.data.clone()
	data.recording = data.recording.frame_slice(0, 1_000_000)
	data.sortings = {'ms3_best': data.sortings['ms3_best'].select_units([2, 7, 11, 14, 21, 23, 24]).frame_slice(0, 1_000_000)}
	mono_sorting_data = MonoSortingData(data, data.sortings['ms3_best'])

	module = RemoveBadUnits("test_rbu", mono_sorting_data, "all")
	assert not os.path.exists(f"{module.logs_folder}/bad_units.html")
	params = module.update_params(PARAMS)
	sorting = module.run(params)
	assert os.path.exists(f"{module.logs_folder}/bad_units.html")

	assert 0 < sorting.get_num_units() < mono_sorting_data.sorting.get_num_units()

	module = RemoveBadUnits("test_rbu_all", mono_sorting_data, "all")
	params = module.update_params({"all": {}})
	sorting = module.run(params)
	assert sorting.get_num_units() == 0
