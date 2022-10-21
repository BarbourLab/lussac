import itertools
import pickle
import numpy as np
import numba
import networkx as nx
import spikeinterface.core as si
from lussac.core.module import MultiSortingsModule


class MergeSortings(MultiSortingsModule):
	"""
	Merges the sortings into a single one.
	"""

	def run(self, params: dict) -> dict[str, si.BaseSorting]:
		# TODO: recenter units between them before comparing them.
		similarity_matrices = self._compute_similarity_matrices(params['similarity']['TODO'])
		graph = self._compute_graph(similarity_matrices, params['similarity']['min_similarity'])
		self.remove_merged_units(graph)

		return self.sortings

	def _compute_similarity_matrices(self, max_time: int) -> dict[str, dict[str, np.ndarray]]:
		"""
		Computes the similarity matrix between all sortings.

		@param max_time: int
			The maximum time difference between spikes to be considered similar.
			Two spikes spaced by exactly max_time are considered coincident.
		@return similarity_matrices: dict[str, dict[str, np.ndarray]]
			The similarity matrices [sorting1, sorting2, similarity_matrix].
		"""

		similarity_matrices = {}
		# TODO: remove duplicated spikes for sorting before to_spike_vector().
		spike_vectors = {name: sorting.to_spike_vector() for name, sorting in self.sortings.items()}
		n_spikes = {name: np.array([len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids]) for name, sorting in self.sortings.items()}

		for name1, sorting1 in self.sortings.items():
			similarity_matrices[name1] = {}
			for name2, sorting2 in self.sortings.items():
				if name1 == name2 or name2 in similarity_matrices[name1]:
					continue
				if name2 not in similarity_matrices:
					similarity_matrices[name2] = {}

				coincidence_matrix = self._compute_coincidence_matrix(spike_vectors[name1]['sample_ind'], spike_vectors[name1]['unit_ind'],
																	  spike_vectors[name2]['sample_ind'], spike_vectors[name2]['unit_ind'], max_time)

				similarity_matrix = coincidence_matrix / np.minimum(n_spikes[name1][:, None], n_spikes[name2])
				expected_matrix = n_spikes[name1][:, None] * n_spikes[name2] * (2*max_time+1) / self.recording.get_num_frames()
				corrected_similarity_matrix = (similarity_matrix - expected_matrix) / (1 - expected_matrix)
				similarity_matrices[name1][name2] = corrected_similarity_matrix
				similarity_matrices[name2][name1] = corrected_similarity_matrix.T

		return similarity_matrices

	@staticmethod
	@numba.jit((numba.int64[:], numba.int64[:], numba.int64[:], numba.int64[:], numba.int32),
			   nopython=True, nogil=True, cache=True)
	def _compute_coincidence_matrix(spike_times1, spike_labels1, spike_times2, spike_labels2, max_time):
		"""
		Computes the number of coincident spikes between all units in two sortings

		@param spike_times1: array[int64] (n_spikes1)
			All the spike timings of the first sorting.
		@param spike_labels1: array[int64] (n_spikes1)
			The unit labels of the first sorting (i.e. unit index of each spike).
		@param spike_times2: array[int64] (n_spikes2)
			All the spike timings of the second sorting.
		@param spike_labels2: array[int64] (n_spikes2)
			The unit labels of the second sorting (i.e. unit index of each spike).
		@param max_time: int32
			The maximum time difference between two spikes to be considered coincident.
			Two spikes spaced by exactly max_time are considered coincident.
		@return coincidence_matrix: array[int64] (n_units1, n_units2)
		"""

		n_units1 = np.max(spike_labels1) + 1
		n_units2 = np.max(spike_labels2) + 1
		coincidence_matrix = np.zeros((n_units1, n_units2), dtype=np.int64)

		start_j = 0
		for i in range(len(spike_times1)):
			for j in range(start_j, len(spike_times2)):
				diff = spike_times1[i] - spike_times2[j]

				if diff > max_time:
					start_j += 1
					continue
				if diff < -max_time:
					break

				coincidence_matrix[spike_labels1[i], spike_labels2[j]] += 1

		return coincidence_matrix

	def _compute_graph(self, similarity_matrices: dict[str, dict[str, np.ndarray]], min_similarity: float) -> nx.Graph:
		"""
		Creates a graph containing all the units from all the sortings,
		and edges between units that are similar.

		@param similarity_matrices: dict[str, dict[str, np.ndarray]]
			The similarity matrices [sorting1, sorting2, similarity_matrix].
		@param min_similarity: float
			The minimal similarity between 2 units to add an edge in the graph.
		@return graph: nx.Graph
			The graph containing the edged for each pair of similar units.
		"""

		graph = nx.Graph()
		for i, (name1, sorting1) in enumerate(self.sortings.items()):
			for j, (name2, sorting2) in enumerate(self.sortings.items()):
				if j <= i:
					continue

				for i, unit_id1 in enumerate(sorting1.unit_ids):
					graph.add_node((name1, unit_id1))
					for j, unit_id2 in enumerate(sorting2.unit_ids):
						graph.add_node((name2, unit_id2))
						if similarity := similarity_matrices[name1][name2][i, j] >= min_similarity:
							graph.add_edge((name1, unit_id1), (name2, unit_id2), similarity=similarity)

		with open(f"{self.logs_folder}/similarity_graph.pkl", 'wb+') as file:
			pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)

		return graph

	def remove_merged_units(self, graph: nx.Graph):
		"""
		Detects and remove merged units from the graph.
		For each connected components (i.e. connected sub-graph communities), look at each node. If a node is connected
		to two nodes coming from the same sorting, then either the original node is a merged unit, or the two nodes are
		split units. In the first case, (under the hypothesis that the 2 neurons are not correlated), the cross-contamination
		between the two nodes should be high (around 100%), whereas in the second case, the cross contamination should be
		low (around 0%). If the cross-contamination crosses a threshold, the original node is labeled as a merged unit
		and all of its edges are removed.

		@param graph: nx.Graph
			The graph containing all the units and connected by their similarity.
		@return:
		"""

		for sub_graph in nx.connected_components_subgraphs(graph):
			for node in sub_graph.nodes:
				for node1, node2 in itertools.combinations(sub_graph.neighbors(node), 2):
					sorting1_name, unit_id1 = node1
					sorting2_name, unit_id2 = node2

					if sorting1_name != sorting2_name:
						continue

					spike_train1 = self.sortings[sorting1_name].get_unit_spike_train(unit_id1)
					spike_train2 = self.sortings[sorting2_name].get_unit_spike_train(unit_id2)
