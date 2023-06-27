import pytest
from lussac.core import MonoSortingData
from lussac.modules import FindPurkinjeCells


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = FindPurkinjeCells("test_fpc_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)

	with pytest.raises(ValueError):
		module.update_params({'cs_max_fr': 10.0, 'ss_min_fr': 8.0})  # cs_max_fr should be lower than ss_min_fr.
	with pytest.raises(ValueError):
		module.update_params({'threshold': -1.0})  # threshold cannot be negative.


def test_find_purkinje_cells(mono_sorting_data: MonoSortingData) -> None:
	assert 'lussac_purkinje' not in mono_sorting_data.sorting.get_property_keys()

	module = FindPurkinjeCells("test_fpc", mono_sorting_data, "all")
	sorting = module.run(module.default_params)

	assert 'lussac_purkinje' in sorting.get_property_keys()
	assert sorting.get_unit_property(66, 'lussac_purkinje') == "66-57"
	assert sorting.get_unit_property(70, 'lussac_purkinje') == "71-70 ; 78-70"
	assert sorting.get_unit_property(71, 'lussac_purkinje') == "71-70"
