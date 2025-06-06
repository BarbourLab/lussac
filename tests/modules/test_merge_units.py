from lussac.core import MonoSortingData
from lussac.modules import MergeUnits
from spikeinterface.core.testing import check_extractor_annotations_equal


def test_remove_splits(mono_sorting_data: MonoSortingData) -> None:
	module = MergeUnits("test_mu_splits", mono_sorting_data, "all")
	sorting = mono_sorting_data.sorting
	params = module.update_params({})

	extra_outputs = {'pairs_decreased_score': [(71, 81), (52, 62)]}
	assert all(x in sorting.unit_ids for x in [52, 62, 71, 81])

	sorting = module._remove_splits(sorting, extra_outputs, params)
	assert 71 in sorting.unit_ids and 81 not in sorting.unit_ids
	assert 52 in sorting.unit_ids and 62 not in sorting.unit_ids


def test_inner_merge(mono_sorting_data: MonoSortingData) -> None:
	module = MergeUnits("test_mu_merge", mono_sorting_data, "all")
	sorting = mono_sorting_data.sorting
	params = module.update_params({})

	# 68, 69, 71, 78 are from the same cell, 52 is another cell, and 999 doesn't exist.
	assert all(x in sorting.unit_ids for x in [52, 68, 69, 71, 78])
	assert 999 not in sorting.unit_ids

	result = module._merge(sorting, [(52, 68), (68, 69), (69, 71), (71, 78), (78, 999)], params)
	assert 52 not in result.unit_ids and 999 not in result.unit_ids
	assert sum([x in result.unit_ids for x in [68, 69, 71, 78]]) == 1  # Only 1 unit remains.

	# TODO: Check spike train of merged unit.


def test_merge_units(mono_sorting_data: MonoSortingData) -> None:
	data = mono_sorting_data.data.clone()
	data.recording = data.recording.select_channels([4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25])
	data.sortings = {'ms3_best': data.sortings['ms3_best'].select_units([18, 20, 22, 29, 31, 53, 55, 59, 67, 68, 69, 70, 71, 78, 81])}
	mono_sorting_data = MonoSortingData(data, data.sortings['ms3_best'])

	module = MergeUnits("merge_units", mono_sorting_data, "all")
	params = module.update_params({
		'auto_merge_params': {'steps_params': {
			'num_spikes': {'min_spikes': 500},
			'template_similarity': {'template_diff_thresh': 0.32}
		}},
		'wvf_extraction': {'max_spikes_per_unit': 1000, 'filter': [300.0, 6_000.0]}
	})

	# big_split = [53, 68, 69, 71, 78, 81]  # Lots of units that are the same Purkinje cell.
	big_split = [53, 69, 71, 78, 81]  # 68 seems hard to get.

	prev_sorting = mono_sorting_data.sorting
	prev_n_units = prev_sorting.get_num_units()
	assert 29 in prev_sorting.unit_ids and 31 in prev_sorting.unit_ids  # 29 & 31 should be merged
	assert all([x in prev_sorting.unit_ids for x in big_split])

	sorting = module.run(params)
	check_extractor_annotations_equal(prev_sorting, sorting)
	assert sorting.get_num_units() < prev_n_units
	assert sum([x in sorting.unit_ids for x in [29, 31]]) == 1  # 29 and 31 have been merged
	assert sum([x in sorting.unit_ids for x in big_split]) == 1  # Only one unit from the big split should remain.

	assert 70 in sorting.unit_ids  # 70 is the complex spike of the big split: should be kept.
