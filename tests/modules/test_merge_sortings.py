import os
import copy
import pickle
import pytest
import networkx as nx
import numpy as np
from lussac.core import LussacData, MultiSortingsData
from lussac.modules import MergeSortings
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
	cross_shifts = {name1: {name2: None for name2 in merge_sortings_module.sortings.keys()} for name1 in merge_sortings_module.sortings.keys()}
	params = {'refractory_period': [0.2, 1.0], 'similarity': {'window': 6}}

	similarity_matrices = merge_sortings_module._compute_similarity_matrices(cross_shifts, params)

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


def test_graph(merge_sortings_module: MergeSortings) -> None:
	G = nx.Graph()
	G.add_node('A', connected=True)
	G.add_node('B', connected=True)
	G.add_node('C', connected=False)
	G.add_edge('A', 'B', similarity=0.9, corr_diff=0.05)

	merge_sortings_module._save_graph(G, "test_save_graph")

	filepath = f"{merge_sortings_module.logs_folder}/test_save_graph.pkl"
	assert os.path.exists(filepath)

	with open(filepath, 'rb') as graph_file:
		graph = pickle.load(graph_file)
		graph_file.close()

	assert graph.number_of_nodes() == 3
	assert graph.number_of_edges() == 1
	assert 'A' in graph.nodes
	assert not graph.nodes['C']['connected']
	assert graph.get_edge_data('A', 'B')['similarity'] == 0.9
	assert graph.get_edge_data('A', 'B')['corr_diff'] == 0.05


def test_remove_merged_units(merge_sortings_module: MergeSortings) -> None:
	# TODO: Use other module to not repeat test_merge_sortings()

	"""logs_file = f"{merge_sortings_module.logs_folder}/merged_units_logs.txt"

	similarity_matrices = merge_sortings_module._compute_similarity_matrices(max_time=5)
	graph = merge_sortings_module._compute_graph(similarity_matrices, min_similarity=0.4)

	assert not os.path.exists(logs_file)
	merge_sortings_module.remove_merged_units(graph)
	assert os.path.exists(logs_file)"""

	# TODO: Check that the units have been removed with correct attributes.


def test_compute_difference(merge_sortings_module: MergeSortings) -> None:
	sortings = merge_sortings_module.sortings
	graph = nx.Graph()
	cross_shifts = {name1: {name2: np.zeros((sorting1.get_num_units(), sorting2.get_num_units()), dtype=np.int64)
					for name2, sorting2 in sortings.items()} for name1, sorting1 in sortings.items()}
	params = merge_sortings_module.update_params({})

	# Test with empty graph
	merge_sortings_module.compute_correlogram_difference(graph, cross_shifts, params['correlogram_validation'])
	merge_sortings_module.compute_waveform_difference(graph, cross_shifts, params['waveform_validation'])

	graph.add_edge(('ks2_low_thresh', 70), ('ms3_best', 71))  # Same unit
	graph.add_edge(('ks2_low_thresh', 64), ('ms3_best', 80))  # Different units
	ss1 = sortings['ks2_low_thresh'].id_to_index(70)
	ss2 = sortings['ms3_best'].id_to_index(71)
	cross_shifts['ks2_low_thresh']['ms3_best'][ss1, ss2] = -1
	merge_sortings_module.compute_correlogram_difference(graph, cross_shifts, params['correlogram_validation'])
	merge_sortings_module.compute_waveform_difference(graph, cross_shifts, params['waveform_validation'])

	assert 'corr_diff' in graph[('ks2_low_thresh', 70)][('ms3_best', 71)]
	assert 'corr_diff' in graph[('ks2_low_thresh', 64)][('ms3_best', 80)]
	assert graph[('ks2_low_thresh', 70)][('ms3_best', 71)]['corr_diff'] < 0.05
	assert graph[('ks2_low_thresh', 64)][('ms3_best', 80)]['corr_diff'] > 0.5

	assert 'temp_diff' in graph[('ks2_low_thresh', 70)][('ms3_best', 71)]
	assert 'temp_diff' in graph[('ks2_low_thresh', 64)][('ms3_best', 80)]
	assert graph[('ks2_low_thresh', 70)][('ms3_best', 71)]['temp_diff'] < 0.10
	assert graph[('ks2_low_thresh', 64)][('ms3_best', 80)]['temp_diff'] > 0.8


def test_merge_sortings_func() -> None:
	# TODO
	pass


@pytest.fixture(scope="function")
def merge_sortings_module(data: LussacData) -> MergeSortings:
	# Copy the dataset with fewer sortings and fewer units to go faster.
	data = data.clone()
	del data.sortings['ms3_low_thresh']
	del data.sortings['ms3_cs']
	del data.sortings['ks2_best']
	del data.sortings['ks2_cs']
	data.sortings['ks2_low_thresh'] = data.sortings['ks2_low_thresh'].select_units([7, 13, 15, 23, 24, 26, 27, 30, 32, 48, 49, 56, 63, 64, 70, 72, 74, 80]).frame_slice(0, 3_000_000)
	data.sortings['ms3_best'] = data.sortings['ms3_best'].select_units([2, 8, 11, 14, 15, 17, 18, 20, 22, 24, 30, 33, 57, 66, 67, 70, 71, 78, 80, 84]).frame_slice(0, 3_000_000)
	data.sortings['ms4_cs'] = data.sortings['ms4_cs'].select_units([0, 5, 7, 8, 9, 17, 19, 20, 23, 25, 32, 53, 56, 58, 59, 62, 69, 70]).frame_slice(0, 3_000_000)

	multi_sortings_data = MultiSortingsData(data, data.sortings)
	return MergeSortings("merge_sortings", multi_sortings_data, "all")
