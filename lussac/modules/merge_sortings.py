import itertools
import pickle
from typing import Any
from overrides import override
import networkx as nx
import numpy as np
from lussac.core.module import MultiSortingsModule
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur
from spikeinterface.curation.curation_tools import find_duplicated_spikes


class MergeSortings(MultiSortingsModule):
	"""
	Merges the sortings into a single one.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'refractory_period': [0.2, 1.0],
			'similarity': {
				'min_similarity': 0.3
			}
		}

	@override
	def run(self, params: dict[str, Any]) -> dict[str, si.BaseSorting]:
		# TODO: recenter units between them before comparing them.
		similarity_matrices = self._compute_similarity_matrices(params['refractory_period'][0])
		graph = self._compute_graph(similarity_matrices, params['similarity']['min_similarity'])
		self.remove_merged_units(graph, params['refractory_period'])
		merged_sorting = self.merge_sortings(graph, params['refractory_period'])

		return {'merged_sorting': merged_sorting}

	def _compute_similarity_matrices(self, censored_period: float) -> dict[str, dict[str, np.ndarray]]:
		"""
		Computes the similarity matrix between all sortings.

		@param censored_period: float
			The maximum time difference between spikes to be considered similar (in ms).
			Two spikes spaced by exactly max_time are considered coincident.
		@return similarity_matrices: dict[str, dict[str, np.ndarray]]
			The similarity matrices [sorting1, sorting2, similarity_matrix].
		"""

		max_time = int(round(censored_period * 1e-3 * self.recording.sampling_frequency))
		similarity_matrices = {}
		spike_vectors = {name: scur.remove_duplicated_spikes(sorting, censored_period).to_spike_vector() for name, sorting in self.sortings.items()}
		n_spikes = {name: np.array(list(sorting.get_total_num_spikes().values())) for name, sorting in self.sortings.items()}

		for name1, sorting1 in self.sortings.items():
			similarity_matrices[name1] = {}
			for name2, sorting2 in self.sortings.items():
				if name1 == name2 or name2 in similarity_matrices[name1]:
					continue
				if name2 not in similarity_matrices:
					similarity_matrices[name2] = {}

				coincidence_matrix = utils.compute_coincidence_matrix_from_vector(spike_vectors[name1], spike_vectors[name2], max_time)

				similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes[name1], n_spikes[name2], max_time)
				similarity_matrices[name1][name2] = similarity_matrix
				similarity_matrices[name2][name1] = similarity_matrix.T

		return similarity_matrices

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
					graph.add_node((name1, unit_id1), connected=False)
					for j, unit_id2 in enumerate(sorting2.unit_ids):
						graph.add_node((name2, unit_id2), connected=False)
						if similarity := similarity_matrices[name1][name2][i, j] >= min_similarity:
							graph.add_edge((name1, unit_id1), (name2, unit_id2), similarity=similarity)
							graph.add_node((name1, unit_id1), connected=True)
							graph.add_node((name2, unit_id2), connected=True)

		with open(f"{self.logs_folder}/similarity_graph.pkl", 'wb+') as file:
			pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)

		return graph

	def remove_merged_units(self, graph: nx.Graph, refractory_period) -> None:
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
		@param refractory_period: float
			The (censored_period, refractory_period) in ms.
		"""

		logs = open(f"{self.logs_folder}/merged_units_logs.txt", 'w+')
		nodes_to_remove = []

		for node in graph.nodes:
			for node1, node2 in itertools.combinations(graph.neighbors(node), 2):
				sorting1_name, unit_id1 = node1
				sorting2_name, unit_id2 = node2

				if sorting1_name != sorting2_name:
					continue

				# TODO: Add checks for spiketrain 1&2 contamination, order? ...
				spike_train1 = self.sortings[sorting1_name].get_unit_spike_train(unit_id1)
				spike_train2 = self.sortings[sorting2_name].get_unit_spike_train(unit_id2)
				cross_cont, p_value = utils.estimate_cross_contamination(spike_train1, spike_train2, refractory_period, limit=0.3)

				logs.write(f"Unit {node} is connected to {node1} and {node2}: cc = {cross_cont:.2%} (p_value={p_value:.3f})\n")
				if p_value > 1e-5:
					continue

				if node not in nodes_to_remove:
					nodes_to_remove.append(node)

		for node in nodes_to_remove:  # Remove node then re-add it no remove all the edges.
			graph.remove_node(node)
			graph.add_node(node, merged=True, connected=False)

		logs.close()

	def merge_sortings(self, graph: nx.Graph, refractory_period) -> si.NpzSortingExtractor:
		"""
		Merges the sortings based on a graph of similar units.

		@param graph: nx.Graph
			The graph containing all the units and connected by their similarity.
		@param refractory_period: float
			The (censored_period, refractory_period) in ms.
		@return merged_sorting: si.NpzSortingExtractor
			The merged sorting.
		"""

		max_units_merge = 2
		k = 2.5
		t_c = int(round(refractory_period[0] * 1e-3 * self.recording.sampling_frequency))
		new_spike_trains = {}
		logs = open(f"{self.logs_folder}/merge_sortings_logs.txt", 'w+')

		for nodes in nx.connected_components(graph):  # For each putative neuron.
			nodes = list(nodes)
			if len(nodes) == 1 and not graph.nodes[nodes[0]]['connected']:
				continue

			new_unit_id = len(new_spike_trains)
			best_score = -100000
			logs.write(f"Making unit {new_unit_id} from {nodes}\n")

			for n_units in range(1, max_units_merge+1):
				for sub_nodes in itertools.combinations(nodes, n_units):
					spike_trains = [self.sortings[name].get_unit_spike_train(unit_id) for name, unit_id in sub_nodes]
					spike_train = np.sort(list(itertools.chain(*spike_trains))).astype(np.int64)
					indices_of_duplicates = find_duplicated_spikes(spike_train, t_c, method="random", seed=np.random.randint(low=0, high=np.iinfo(np.int32).max))
					spike_train = np.delete(spike_train, indices_of_duplicates)

					f = len(spike_train) * self.sampling_f / self.recording.get_num_frames()
					C = utils.estimate_contamination(spike_train, refractory_period)
					score = f * (1 - (k+1)*C)
					logs.write(f"\t- Score = {score:.2f}\t[{sub_nodes}]")

					if score > best_score:
						best_score = score
						new_spike_trains[new_unit_id] = spike_train

		logs.close()

		sorting = si.NumpySorting.from_dict(new_spike_trains, self.sampling_f)
		filename = f"{self.logs_folder}/merged_sorting.npz"
		si.NpzSortingExtractor.write_sorting(sorting, filename)

		return si.NpzSortingExtractor(filename)
