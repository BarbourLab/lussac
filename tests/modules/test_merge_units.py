from lussac.core.lussac_data import MonoSortingData
from lussac.modules.merge_units import MergeUnits


def test_merge_units(mono_sorting_data: MonoSortingData) -> None:
	# Take a subset of units to accelerate the test.

	module = MergeUnits("merge_units", mono_sorting_data, "all")
	params = module.update_params({})

	sorting = module.run(params)
