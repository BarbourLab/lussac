from lussac.core import MonoSortingData
from lussac.modules import MergeUnits
from spikeinterface.core.testing import check_extractor_annotations_equal


def test_merge_units(mono_sorting_data: MonoSortingData) -> None:
	# TODO: Take a subset of units to accelerate the test.

	module = MergeUnits("merge_units", mono_sorting_data, "all")
	params = module.update_params({})

	big_split = [53, 68, 69, 71, 78, 81]  # Lots of units that are the same Purkinje cell.

	prev_sorting = mono_sorting_data.sorting
	prev_n_units = prev_sorting.get_num_units()
	assert 29 in prev_sorting.unit_ids and 31 in prev_sorting.unit_ids  # 29 & 31 should be merged
	assert all([x in prev_sorting.unit_ids for x in big_split])

	sorting = module.run(params)
	check_extractor_annotations_equal(prev_sorting, sorting)
	assert sorting.get_num_units() < prev_n_units
	assert sum([x in sorting.unit_ids for x in [29, 31]]) == 1  # 29 and 31 have been merged
	assert sum([x in sorting.unit_ids for x in big_split]) == 1  # Only one unit from the big split should remain.


def test_remove_splits(mono_sorting_data: MonoSortingData) -> None:
	module = MergeUnits("test", mono_sorting_data, "all")
	sorting = mono_sorting_data.sorting
	params = module.update_params({})

	extra_outputs = {'pairs_decreased_score': [(71, 81), (52, 62)]}
	assert all(x in sorting.unit_ids for x in [52, 62, 71, 81])

	sorting = module._remove_splits(sorting, extra_outputs, params)
	assert 71 in sorting.unit_ids and 81 not in sorting.unit_ids
	assert 52 in sorting.unit_ids and 62 not in sorting.unit_ids


def test_inner_merge(mono_sorting_data: MonoSortingData) -> None:
	pass  # TODO
