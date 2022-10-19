import numpy as np
from lussac.core.lussac_data import MonoSortingData
from lussac.modules.units_categorization import UnitsCategorization


def test_units_categorization(mono_sorting_data: MonoSortingData) -> None:
	params = {"CS": {
		"frequency": {
			"min": 0.2,
			"max": 5.0
		},
		"ISI_portion": {
			"range": [8.0, 35.0],
			"max": 0.012
		}
	}}

	module = UnitsCategorization("test_categorization", mono_sorting_data, "all")
	sorting = module.run(params)

	assert np.all(sorting.unit_ids[sorting.get_property("lussac_category") == "CS"] == (2, 5, 8, 51, 56, 57, 70))

	sorting = module.run({"clear": {}})
	assert sorting.get_property("lussac_category") is None
