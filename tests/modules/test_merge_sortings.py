import os
import copy
import pickle
import pytest
import networkx as nx
import numpy as np
from lussac.core.lussac_data import LussacData, MultiSortingsData
from lussac.modules.merge_sortings import MergeSortings
import spikeinterface.core as si


def test_default_params(merge_sortings_module: MergeSortings) -> None:
	assert isinstance(merge_sortings_module.default_params, dict)


def test_merge_sortings(merge_sortings_module: MergeSortings) -> None:
	assert not os.path.exists(f"{merge_sortings_module.logs_folder}/merge_sortings_logs.txt")

	params = {'refractory_period': [0.2, 1.0], 'similarity': {'min_similarity': 0.4}}
	params = merge_sortings_module.update_params(params)
	sortings = merge_sortings_module.run(params)

	assert len(sortings) == 1
	assert 'merged_sorting' in sortings
	assert sortings['merged_sorting'].get_annotation('name') == "merged_sorting"
	assert sortings['merged_sorting'].get_num_units() > 10
	assert os.path.exists(f"{merge_sortings_module.logs_folder}/merge_sortings_logs.txt")


def test_compute_similarity_matrices(merge_sortings_module: MergeSortings) -> None:
	similarity_matrices = merge_sortings_module._compute_similarity_matrices(0.2)

	assert 'ks2_low_thresh' in similarity_matrices
	assert 'ms4_cs' in similarity_matrices['ms3_best']
	assert np.max(similarity_matrices['ks2_low_thresh']['ms3_best']) <= 1.0


def test_compute_graph(data: LussacData) -> None:
	sortings = {
		'1': si.NumpySorting.from_dict({0: np.array([0]), 1: np.array([0]), 2: np.array([0])}, sampling_frequency=30000),
		'2': si.NumpySorting.from_dict({0: np.array([0]), 1: np.array([0]), 2: np.array([0])}, sampling_frequency=30000),
		'3': si.NumpySorting.from_dict({0: np.array([0]), 1: np.array([0])}, sampling_frequency=30000)
	}
	multi_sortings_data = MultiSortingsData(data, sortings)
	module = MergeSortings("merge_sortings", multi_sortings_data, "all")

	similarity_matrices = {
		'1': {
			'2': np.array([[0.9, -0.05, 0.2], [0.02, 0.95, 0.6], [-0.03, 0.0, 0.01]]),
			'3': np.array([[1.0, 0.03], [0.0, -0.6], [0.0, 0.0]])
		},
		'2': {
			'1': np.array([[0.9, 0.02, -0.03], [-0.05, 0.95, 0.0], [0.2, 0.6, 0.01]]),
			'3': np.array([[0.95, -0.01], [0.04, -0.04], [0.4, 0.3]])
		},
		'3': {
			'1': np.array([[1.0, 0.0, 0.0], [0.03, -0.6, 0.0]]),
			'2': np.array([[0.95, 0.04, 0.4], [-0.01, -0.04, 0.3]])
		}
	}

	graph = module._compute_graph(similarity_matrices, min_similarity=0.4)

	assert graph.number_of_nodes() == 8
	assert graph.number_of_edges() == 6
	assert graph.has_edge(('1', 0), ('2', 0))
	assert graph.has_edge(('1', 1), ('2', 1))
	assert graph.has_edge(('2', 2), ('3', 0))
	assert not graph.has_edge(('1', 2), ('2', 2))

	with open(f"{module.logs_folder}/similarity_graph.pkl", 'rb') as file:
		graph_loaded = pickle.load(file)
		assert nx.is_isomorphic(graph, graph_loaded)


# TODO: Test save_graph.


def test_remove_merged_units(merge_sortings_module: MergeSortings) -> None:
	# TODO: Use other module to not repeat test_merge_sortings()

	"""logs_file = f"{merge_sortings_module.logs_folder}/merged_units_logs.txt"

	similarity_matrices = merge_sortings_module._compute_similarity_matrices(max_time=5)
	graph = merge_sortings_module._compute_graph(similarity_matrices, min_similarity=0.4)

	assert not os.path.exists(logs_file)
	merge_sortings_module.remove_merged_units(graph)
	assert os.path.exists(logs_file)"""

	# TODO: Check that the units have been removed with correct attributes.


# TODO: test compute_correlogram_difference


def test_merge_sortings_func() -> None:
	# TODO
	pass


@pytest.fixture(scope="function")
def merge_sortings_module(data: LussacData) -> MergeSortings:
	data = copy.deepcopy(data)
	del data.sortings['ms3_low_thresh']
	del data.sortings['ms3_cs']
	del data.sortings['ks2_best']
	del data.sortings['ks2_cs']

	# TODO: Maybe remove some units to accelerate the test.

	multi_sortings_data = MultiSortingsData(data, data.sortings)
	return MergeSortings("merge_sortings", multi_sortings_data, "all")
