import copy
import pytest
import numpy as np
import spikeinterface.core as si
from lussac.core.lussac_data import LussacData, MultiSortingsData
from lussac.modules.merge_sortings import MergeSortings


def test_compute_coincidence_matrix() -> None:
	sorting1 = si.NumpySorting.from_dict({0: np.array([18, 163, 622, 1197]), 1: np.array([161, 300, 894])}, sampling_frequency=30000)
	sorting2 = si.NumpySorting.from_dict({0: np.array([120, 298, 303, 628]), 1: np.array([84, 532, 1092])}, sampling_frequency=30000)
	spike_vector1 = sorting1.to_spike_vector()
	spike_vector2 = sorting2.to_spike_vector()

	coincidence_matrix = MergeSortings._compute_coincidence_matrix(spike_vector1['sample_ind'], spike_vector1['unit_ind'],
																   spike_vector2['sample_ind'], spike_vector2['unit_ind'], 8)

	assert coincidence_matrix[0, 0] == 1
	assert coincidence_matrix[0, 1] == 0
	assert coincidence_matrix[1, 0] == 2


def test_compute_similarity_matrices(merge_sortings_module: MergeSortings):
	similarity_matrices = merge_sortings_module._compute_similarity_matrices(6)

	assert 'ks2_low_thresh' in similarity_matrices
	assert 'ms4_cs' in similarity_matrices['ms3_best']


@pytest.fixture(scope="function")
def merge_sortings_module(data: LussacData) -> MergeSortings:
	data = copy.deepcopy(data)
	del data.sortings['ms3_low_thresh']
	del data.sortings['ms3_cs']
	del data.sortings['ks2_best']
	del data.sortings['ks2_cs']

	# TODO: Maybe remove some units to accelerate the test.

	multi_sortings_data = MultiSortingsData(data, data.sortings)
	return MergeSortings("merge_sortings", multi_sortings_data, "all", "")
