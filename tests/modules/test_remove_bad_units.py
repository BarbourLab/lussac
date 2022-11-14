import copy
import os
from lussac.core.lussac_data import MonoSortingData
from lussac.modules.remove_bad_units import RemoveBadUnits


params = {
	"firing_rate": {
		"min": 0.5
	},
	"contamination": {
		"refractory_period": [0.5, 1.0],
		"max": 0.25
	},
	"amplitude": {
		"wvf_extraction": {
			"ms_before": 1.0,
			"ms_after": 1.0,
			"max_spikes_per_unit": 10
		},
		"filter": {
			"band": [200, 5000]
		},
		"min": 20
	},
	"SNR": {
		"wvf_extraction": {
			"ms_before": 1.0,
			"ms_after": 1.0,
			"max_spikes_per_unit": 10
		},
		"filter": {
			"band": [300, 6000]
		},
		"min": 1.2
	},
	"amplitude_std": {
		"wvf_extraction": {
			"ms_before": 1.0,
			"ms_after": 1.0,
		},
		"filter": {
			"band": [200, 5000]
		},
		"max": 140
	}
}


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = RemoveBadUnits("test_rbu_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_remove_bad_units(mono_sorting_data: MonoSortingData) -> None:
	data = copy.deepcopy(mono_sorting_data)
	data.sorting = data.sorting.select_units([2, 7, 11, 14, 21, 23, 24])

	module = RemoveBadUnits("test_rbu", data, "all")
	assert not os.path.exists(f"{module.logs_folder}/bad_units.html")
	sorting = module.run(params)
	assert os.path.exists(f"{module.logs_folder}/bad_units.html")

	assert 0 < sorting.get_num_units() < data.sorting.get_num_units()

	module = RemoveBadUnits("test_rbu_all", data, "all")
	sorting = module.run({"all": {}})
	assert sorting.get_num_units() == 0
