import numpy as np
from lussac.core import MonoSortingData
from lussac.modules import UnitsCategorization


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = UnitsCategorization("test_categorization_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_units_categorization(mono_sorting_data: MonoSortingData) -> None:
	params = {"CS": {
		"firing_rate": {
			'min': 0.2,
			'max': 5.0
		},
		"ISI_portion": {
			'range': [8.0, 35.0],
			'max': 0.012
		},
		"sd_ratio": {
			'min': 0.1,
			'max': 5.0
		}
	}}

	module = UnitsCategorization("test_categorization", mono_sorting_data, "all")
	params = module.update_params(params)
	sorting = module.run(params)

	assert np.all(sorting.unit_ids[sorting.get_property("lussac_category") == "CS"] == (2, 5, 8, 51, 56, 57, 70))

	module.analyzer = None
	params = module.update_params({"clear": {}})
	sorting = module.run(params)
	assert sorting.get_property("lussac_category") is None
