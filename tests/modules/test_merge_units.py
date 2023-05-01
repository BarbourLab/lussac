from lussac.core import MonoSortingData
from lussac.modules import MergeUnits
from spikeinterface.core.testing import check_extractor_annotations_equal


def test_merge_units(mono_sorting_data: MonoSortingData) -> None:
	# TODO: Take a subset of units to accelerate the test.

	module = MergeUnits("merge_units", mono_sorting_data, "all")
	params = module.update_params({})

	prev_sorting = mono_sorting_data.sorting
	prev_n_units = prev_sorting.get_num_units()
	assert 29 in prev_sorting.unit_ids and 31 in prev_sorting.unit_ids  # 29 & 31 should be merged

	sorting = module.run(params)
	check_extractor_annotations_equal(prev_sorting, sorting)
	assert sorting.get_num_units() < prev_n_units
	assert 29 not in sorting.unit_ids and 31 not in sorting.unit_ids
