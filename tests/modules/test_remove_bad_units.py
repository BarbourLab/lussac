import copy
import pytest
import numpy as np
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


def test_remove_bad_units(mono_sorting_data: MonoSortingData) -> None:
	data = copy.deepcopy(mono_sorting_data)
	data.sorting = data.sorting.select_units([2, 7, 11, 14, 21, 23, 24])

	module = RemoveBadUnits("test_rbu", data, "all")
	sorting = module.run(params)

	assert 0 < sorting.get_num_units() < data.sorting.get_num_units()
	# TODO: Test that plots are generated.

	module = RemoveBadUnits("test_rbu_all", data, "all")
	sorting = module.run({"all": {}})
	assert sorting.get_num_units() == 0


def test_get_units_attribute(mono_sorting_data: MonoSortingData) -> None:
	data = copy.deepcopy(mono_sorting_data)
	data.sorting = data.sorting.select_units([0, 1, 2, 3, 4])
	module = RemoveBadUnits("test_rbu_get_units_attribute", data, "all")

	frequencies = module.get_units_attribute_arr("firing_rate", params['firing_rate'])
	assert isinstance(frequencies, np.ndarray)
	assert frequencies.shape == (data.sorting.get_num_units(), )
	assert abs(frequencies[0] - 22.978) < 0.01

	contamination = module.get_units_attribute_arr("contamination", params["contamination"])
	assert isinstance(contamination, np.ndarray)
	assert contamination.shape == (data.sorting.get_num_units(), )
	assert contamination[0] >= 1.0

	amplitude = module.get_units_attribute_arr("amplitude", params["amplitude"])
	assert isinstance(amplitude, np.ndarray)
	assert amplitude.shape == (data.sorting.get_num_units(), )

	amplitude_std = module.get_units_attribute_arr("amplitude_std", params["amplitude_std"])
	assert isinstance(amplitude_std, np.ndarray)
	assert amplitude_std.shape == (data.sorting.get_num_units(), )

	with pytest.raises(ValueError):
		module.get_units_attribute("test", {})
