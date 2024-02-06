import itertools
import math
import pickle
from typing import Any
from overrides import override
import networkx as nx
import numpy as np
from lussac.core import MultiSortingsModule
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur
from spikeinterface.curation.curation_tools import find_duplicated_spikes
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm


class MergeSortings(MultiSortingsModule):
	"""
	Merges the sortings into a single one.
	"""

	aggregated_wvf_extractor: si.WaveformExtractor  # Waveform extractor for all analyses.

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'refractory_period': [0.2, 1.0],
			'max_shift': 1.33,
			'require_multiple_sortings_match': True,
			'max_units_merge': 1,
			'similarity': {
				'min_similarity': 0.3,
				'window': 0.2
			},
			'correlogram_validation': {
				'max_time': 70.0,
				'gaussian_std': 0.6,
				'gaussian_truncate': 5.0
			},
			'waveform_validation': {
				'wvf_extraction': {
					'ms_before': 1.0,
					'ms_after': 2.0,
					'max_spikes_per_unit': 1_000,
					'filter': [250.0, 6_000.0]
				},
				'num_channels': 5
			},
			'merge_check': {
				'cross_cont_threshold': 0.10
			},
			'clean_edges': {
				'template_diff_threshold': 0.10,
				'corr_diff_threshold': 0.12,
				'cross_cont_threshold': 0.06
			}
		}

	@override
	def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
		params = super().update_params(params)

		params['similarity']['window'] = int(round(params['similarity']['window'] * 1e-3 * self.sampling_f))
		params['correlogram_validation']['max_time'] = int(round(params['correlogram_validation']['max_time'] * 1e-3 * self.sampling_f))
		params['correlogram_validation']['gaussian_std'] = params['correlogram_validation']['gaussian_std'] * 1e-3 * self.sampling_f
		params['correlogram_validation']['censored_period'] = params['similarity']['window']
		params['waveform_validation']['wvf_extraction']['ms_before'] += params['max_shift']
		params['waveform_validation']['wvf_extraction']['ms_after']  += params['max_shift']
		params['max_shift'] = int(round(params['max_shift'] * 1e-3 * self.sampling_f))

		return params

	@override
	def run(self, params: dict[str, Any]) -> dict[str, si.BaseSorting]:
		self.data.sortings = {name: sorting.remove_empty_units() for name, sorting in self.sortings.items()}

		self.aggregated_wvf_extractor = self.extract_waveforms(sparse=False, **params['waveform_validation']['wvf_extraction'])
		cross_shifts = self.compute_cross_shifts(params['max_shift'])

		similarity_matrices = self._compute_similarity_matrices(cross_shifts, params)
		graph = self._compute_graph(similarity_matrices, params)
		self.compute_correlogram_difference(graph, cross_shifts, params['correlogram_validation'])
		self.compute_waveform_difference(graph, cross_shifts, params)

		if params['merge_check']:
			self.remove_merged_units(graph, cross_shifts, params['refractory_period'], params['merge_check'])
		self.clean_edges(graph, cross_shifts, params)
		if len(self.sortings) >= 4:
			self.separate_communities(graph)
		self._save_graph(graph, "final_graph")

		merged_sorting = self.merge_sortings(graph, params)

		return {'merged_sorting': merged_sorting}

	def compute_cross_shifts(self, max_shift: int) -> dict[str, dict[str, np.ndarray]]:
		"""
		Computes the cross shifts between units of different sortings.
		This is important when comparing units to make sure they are aligned properly between them.
		If two units don't come from the same neuron, the computed cross shift should be 0.
		A cross shift of -2 means the first unit is generally 2 samples earlier than the second unit.

		@param max_shift: int
			The maximum shift to consider (in samples).
		@return: dict[str, dict[str, np.ndarray]]
			For each pair of analyses, the cross-shifts matrix.
		"""

		spike_vectors = {name: sorting.to_spike_vector() for name, sorting in self.sortings.items()}
		cross_shifts = {name1: {} for name1 in self.sortings.keys()}

		for i, (name1, sorting1) in enumerate(self.sortings.items()):
			for j, (name2, sorting2) in enumerate(self.sortings.items()):
				if i >= j:
					continue

				shifts = utils.compute_cross_shift_from_vector(spike_vectors[name1], spike_vectors[name2], max_shift)

				cross_shifts[name1][name2] = shifts
				cross_shifts[name2][name1] = -shifts.T

		return cross_shifts

	def _compute_similarity_matrices(self, cross_shifts: dict[str, dict[str, np.ndarray | None]], params: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
		"""
		Computes the similarity matrix between all sortings.

		@param cross_shifts: dict[str, dict[str, np.ndarray | None]]
			The cross-shifts between units.
			If None, will default to no shift.
		@param params: dict
			The parameters of the merge_sorting module.
		@return similarity_matrices: dict[str, dict[str, np.ndarray]]
			The similarity matrices [sorting1, sorting2, similarity_matrix].
		"""
		window = params['similarity']['window']
		censored_period = params['refractory_period'][0]

		similarity_matrices = {}
		spike_vectors = {name: scur.remove_duplicated_spikes(sorting, censored_period).to_spike_vector() for name, sorting in self.sortings.items()}
		n_spikes = {name: np.array(list(sorting.count_num_spikes_per_unit().values())) for name, sorting in self.sortings.items()}

		for name1, sorting1 in self.sortings.items():
			similarity_matrices[name1] = {}
			for name2, sorting2 in self.sortings.items():
				if name1 == name2 or name2 in similarity_matrices[name1]:
					continue
				if name2 not in similarity_matrices:
					similarity_matrices[name2] = {}

				coincidence_matrix = utils.compute_coincidence_matrix_from_vector(spike_vectors[name1], spike_vectors[name2], window, cross_shifts[name1][name2])

				similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes[name1], n_spikes[name2], window)
				similarity_matrices[name1][name2] = similarity_matrix
				similarity_matrices[name2][name1] = similarity_matrix.T

		return similarity_matrices

	def _compute_graph(self, similarity_matrices: dict[str, dict[str, np.ndarray]], params: dict) -> nx.Graph:
		"""
		Creates a graph containing all the units from all the sortings,
		and edges between units that are similar.

		@param similarity_matrices: dict[str, dict[str, np.ndarray]]
			The similarity matrices [sorting1, sorting2, similarity_matrix].
		@param params: dict
			The parameters of the merge_sorting module.
		@return graph: nx.Graph
			The graph containing the edged for each pair of similar units.
		"""

		censored_period, refractory_period = params['refractory_period']

		# Populating the graph with all the nodes (i.e. all the units) with properties.
		graph = nx.Graph()
		spost.compute_spike_amplitudes(self.aggregated_wvf_extractor, peak_sign="both", return_scaled=self.recording.has_scaled())
		contamination, _ = sqm.compute_refrac_period_violations(self.aggregated_wvf_extractor, refractory_period_ms=refractory_period, censored_period_ms=censored_period)
		sd_ratio = sqm.compute_sd_ratio(self.aggregated_wvf_extractor)
		snrs = sqm.compute_snrs(self.aggregated_wvf_extractor, peak_sign="both", peak_mode="extremum")

		for (name, sorting) in self.sortings.items():
			for unit_id in sorting.unit_ids:
				new_unit_id = self.aggregated_wvf_extractor.renamed_unit_ids[name][unit_id]

				attr = {key: sorting.get_unit_property(unit_id, key) for key in sorting.get_property_keys() if key.startswith('gt_')}
				attr['contamination'] = contamination[new_unit_id]
				attr['sd_ratio'] = sd_ratio[new_unit_id]
				attr['SNR'] = snrs[new_unit_id]
				graph.add_node((name, unit_id), **attr)

		# Connecting nodes using the similarity.
		for i, (name1, sorting1) in enumerate(self.sortings.items()):
			for j, (name2, sorting2) in enumerate(self.sortings.items()):
				if j <= i:
					continue

				for unit_ind1, unit_id1 in enumerate(sorting1.unit_ids):
					for unit_ind2, unit_id2 in enumerate(sorting2.unit_ids):
						if (similarity := similarity_matrices[name1][name2][unit_ind1, unit_ind2]) >= params['similarity']['min_similarity']:
							graph.add_edge((name1, unit_id1), (name2, unit_id2), similarity=similarity)

		if params['require_multiple_sortings_match']:
			for node in dict(graph.nodes):
				if len(graph.edges(node)) == 0:
					graph.remove_node(node)

		self._save_graph(graph, "similarity_graph")
		return graph

	def _save_graph(self, graph: nx.Graph, name: str) -> None:
		"""
		Saves the current state of the graph in the pickle format.

		@param graph: nx.Graph
			The graph to save.
		@param name: str
			The name of the file (without the .pkl extension)
		"""

		with open(f"{self.logs_folder}/{name}.pkl", 'wb+') as file:
			pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)

	def remove_merged_units(self, graph: nx.Graph, cross_shifts: dict[str, dict[str, np.ndarray]], refractory_period, params: dict[str, Any]) -> None:
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
		@param cross_shifts: dict[str, dict[str, np.ndarray]]
			The cross-shifts between units.
		@param refractory_period: float
			The (censored_period, refractory_period) in ms.
		@param params: dict[str, Any]
			The parameters for the remove_merge function.
		"""

		logs = open(f"{self.logs_folder}/merged_units_logs.txt", 'w+')
		nodes_to_remove = []

		for node in graph.nodes:
			for node1, node2 in itertools.combinations(graph.neighbors(node), 2):
				sorting_name, unit_id = node
				sorting1_name, unit_id1 = node1
				sorting2_name, unit_id2 = node2

				unit_ind = self.sortings[sorting_name].id_to_index(unit_id)
				unit_ind1 = self.sortings[sorting1_name].id_to_index(unit_id1)
				unit_ind2 = self.sortings[sorting2_name].id_to_index(unit_id2)

				if sorting1_name != sorting2_name:
					continue

				spike_train = self.sortings[sorting_name].get_unit_spike_train(unit_id)
				spike_train1 = self.sortings[sorting1_name].get_unit_spike_train(unit_id1) + cross_shifts[sorting_name][sorting1_name][unit_ind, unit_ind1]
				spike_train2 = self.sortings[sorting2_name].get_unit_spike_train(unit_id2) + cross_shifts[sorting_name][sorting2_name][unit_ind, unit_ind2]
				C = graph.nodes[node]['contamination']
				C1 = graph.nodes[node1]['contamination']
				C2 = graph.nodes[node2]['contamination']
				sd = graph.nodes[node]['sd_ratio']
				sd1 = graph.nodes[node1]['sd_ratio']
				sd2 = graph.nodes[node2]['sd_ratio']
				if C2 < C1:
					spike_train1, spike_train2 = spike_train2, spike_train1
				cross_cont, p_value = utils.estimate_cross_contamination(spike_train1, spike_train2, refractory_period, limit=params['cross_cont_threshold'])

				logs.write(f"\nUnit {node} is connected to {node1} and {node2}:\n")
				logs.write(f"\tcross-cont = {cross_cont:.2%} (p_value={p_value:.2e})\n")
				logs.write(f"\tC = {C:.1%} ; C1 = {C1:.1%} ; C2 = {C2:.1%}\n")
				logs.write(f"\tsd = {sd:.3f} ; sd1 = {sd1:.3f} ; sd2 = {sd2:.3f}\n")
				if p_value > 5e-3:  # No problem detected, node1 and node2 are probably just a split.
					continue

				cases_removed = np.array([  # If at least one is true, 'node' is removed
					p_value < 1e-8 and C > max(C1, C2),
					p_value < 5e-3 and sd > max(sd1, sd2) and C > max(C1, C2)
				], dtype=bool)
				if np.any(cases_removed):
					nodes_to_remove.append(node)
					break  # Don't need to check this node again.
				else:  # Couldn't definitely say that 'node' is a merged unit. Check whether 'node1' or 'node2' is bad.
					cc1, p1 = utils.estimate_cross_contamination(spike_train, spike_train1, (0.4, 1.8), limit=0.10)
					cc2, p2 = utils.estimate_cross_contamination(spike_train, spike_train2, (0.4, 1.8), limit=0.10)
		
					if p1 < 5e-3 and cc2 < 0.1 and C1 > C + 0.03:
						nodes_to_remove.append(node1)
					elif p2 < 5e-3 and cc1 < 0.1 and C2 > C + 0.03:
						nodes_to_remove.append(node2)

		if len(nodes_to_remove) > 0:
			logs.write("\nRemoved units:\n")
			for node in set(nodes_to_remove):
				graph.remove_node(node)

				sorting_name, unit_id = node
				label = f" -- {self.sortings[sorting_name].get_unit_property(unit_id, 'gt_label')}" if 'gt_label' in self.sortings[sorting_name].get_property_keys() else ''
				logs.write(f"\t- {node}{label}\n")

		logs.close()

	def compute_correlogram_difference(self, graph: nx.Graph, cross_shifts: dict[str, dict[str, np.ndarray]], params: dict[str, Any]) -> None:
		"""
		Computes the correlogram difference for each edge of the graph, and adds it as an attribute to the edge.

		@param graph: nx.Graph
			The graph containing all the units and connected by their similarity.
		@param cross_shifts: dict[str, dict[str, np.ndarray]]
			The cross-shifts between units.
		@param params: dict
			The parameters for the correlogram difference.
		"""

		N = math.ceil(params['gaussian_std'] * params['gaussian_truncate'])
		gaussian = utils.gaussian_pdf(np.arange(-N, N+1), 0.0, params['gaussian_std'])

		for nodes in nx.connected_components(graph):  # For each community.
			nodes = list(nodes)
			if len(nodes) <= 1:
				continue

			subgraph = graph.subgraph(nodes)

			auto_correlograms = {}
			for node in nodes:
				sorting_name, unit_id = node
				spike_train = self.sortings[sorting_name].get_unit_spike_train(unit_id)
				auto_corr = spost.compute_autocorrelogram_from_spiketrain(spike_train, window_size=params['max_time'], bin_size=1)
				auto_correlograms[node] = np.convolve(auto_corr, gaussian, mode="same")

			for node1, node2 in subgraph.edges:
				sorting1_name, unit_id1 = node1
				sorting2_name, unit_id2 = node2
				unit_ind1 = self.sortings[sorting1_name].id_to_index(unit_id1)
				unit_ind2 = self.sortings[sorting2_name].id_to_index(unit_id2)

				shift = cross_shifts[sorting1_name][sorting2_name][unit_ind1, unit_ind2]
				spike_train1 = self.sortings[sorting1_name].get_unit_spike_train(unit_id1)
				spike_train2 = self.sortings[sorting2_name].get_unit_spike_train(unit_id2) + shift
				cross_corr = spost.compute_crosscorrelogram_from_spiketrain(spike_train1, spike_train2, window_size=params['max_time'], bin_size=1)

				middle = len(cross_corr) // 2
				cross_corr[middle-params['censored_period']:middle+params['censored_period']] = 0  # Remove duplicates before filtering.
				cross_corr = np.convolve(cross_corr, gaussian, mode="same")

				corr_diff = utils.compute_correlogram_difference(auto_correlograms[node1], auto_correlograms[node2], cross_corr, len(spike_train1), len(spike_train2))
				graph[node1][node2]['corr_diff'] = corr_diff

	def compute_waveform_difference(self, graph: nx.Graph, cross_shifts: dict[str, dict[str, np.ndarray]], params: dict[str, Any]) -> None:
		"""
		Computes the waveform difference for each edge of the graph, and adds it as an attribute to the edge.

		@param graph: nx.Graph
			The graph containing all the units and connected by their similarity.
		@param cross_shifts: dict[str, dict[str, np.ndarray]]
			The cross-shifts between units.
		@param params: dict
			The parameters for the remove_merge function.
		"""

		n_channels = params['waveform_validation']['num_channels']
		margin = params['max_shift']

		for node1, node2 in graph.edges:
			sorting1_name, unit_id1 = node1
			sorting2_name, unit_id2 = node2
			unit_idx1 = self.sortings[sorting1_name].id_to_index(unit_id1)
			unit_idx2 = self.sortings[sorting2_name].id_to_index(unit_id2)
			new_unit_id1 = self.aggregated_wvf_extractor.renamed_unit_ids[sorting1_name][unit_id1]
			new_unit_id2 = self.aggregated_wvf_extractor.renamed_unit_ids[sorting2_name][unit_id2]

			template1 = self.aggregated_wvf_extractor.get_template(new_unit_id1)
			template2 = self.aggregated_wvf_extractor.get_template(new_unit_id2)
			channel_indices = np.argsort(np.max(np.abs(template1) + np.abs(template2), axis=0))[:-n_channels-1:-1]

			shift = cross_shifts[sorting1_name][sorting2_name][unit_idx1, unit_idx2]
			template1 = template1[margin:-margin, channel_indices]
			template2 = template2[margin+shift:shift-margin, channel_indices]

			template_diff = np.sum(np.abs(template1 - template2)) / np.sum(np.abs(template1) + np.abs(template2))
			graph[node1][node2]['temp_diff'] = template_diff

	def clean_edges(self, graph: nx.Graph, cross_shifts: dict[str, dict[str, np.ndarray]], params: dict) -> None:
		"""
		Checks for edges that are bad (based on cross-contamination, template/correlogram difference)
		If an edge is bad, tries to find if one of the nodes is bad, and removes it
		If it cannot find a bad node, then it simply removes the edge.
		Nodes left alone (i.e. not connected) by this process are removed.

		@param graph: nx.Graph
			The graph containing all the units and connected by their similarity.
		@param cross_shifts: dict[str, dict[str, np.ndarray]]
			The cross-shifts between units.
		@param params: dict
			The parameters of the merge_sorting module.
		"""

		logs = open(f"{self.logs_folder}/clean_edges_logs.txt", 'w+')
		sd_threshold = 0.20  # TODO: Make these parameters.
		C_threshold = 0.08
		refractory_period = params['refractory_period']
		params = params['clean_edges']

		nodes_to_remove = []
		edges_to_remove = []

		for node1, node2, data in list(graph.edges(data=True)):
			sorting_name1, unit_id1 = node1
			sorting_name2, unit_id2 = node2

			unit_ind1 = self.sortings[sorting_name1].id_to_index(unit_id1)
			unit_ind2 = self.sortings[sorting_name2].id_to_index(unit_id2)

			C1 = graph.nodes[node1]['contamination']
			C2 = graph.nodes[node2]['contamination']

			sd1 = graph.nodes[node1]['sd_ratio']
			sd2 = graph.nodes[node2]['sd_ratio']

			spike_train1 = self.sortings[sorting_name1].get_unit_spike_train(unit_id1)
			spike_train2 = self.sortings[sorting_name2].get_unit_spike_train(unit_id2) + cross_shifts[sorting_name1][sorting_name2][unit_ind1, unit_ind2]
			if C2 < C1:
				spike_train1, spike_train2 = spike_train2, spike_train1

			cross_cont, p_value = utils.estimate_cross_contamination(spike_train1, spike_train2, refractory_period, limit=params['cross_cont_threshold'])

			problem_cases = [
				p_value < 5e-3 and data['temp_diff'] > params['template_diff_threshold'] and data['corr_diff'] > params['corr_diff_threshold'],
				p_value < 0.80 and data['similarity'] < 0.60 and data['temp_diff'] > 0.05 and data['corr_diff'] > 0.05
			]

			if not np.any(problem_cases):
				continue

			# From this point on, the edge is treated as problematic.
			logs.write(f"\nEdge {node1} -- {node2} is problematic:\n")
			logs.write(f"\t- {node1} - {graph.nodes[node1]}\n")
			logs.write(f"\t- {node2} - {graph.nodes[node2]}\n")
			logs.write(f"\t- cross-cont = {cross_cont:.2%} (p={p_value:.2e})\n")
			logs.write(f"\t- edge data: {data}\n")

			if sd1 - sd2 > (C2 - C1) * sd_threshold/C_threshold + sd_threshold and sd1 > 1.05:  # node 1 is problematic
				logs.write(f"\t=> Removing node {node1}\n")
				nodes_to_remove.append(node1)
			elif sd2 - sd1 > (C1 - C2) * sd_threshold/C_threshold + sd_threshold and sd2 > 1.05:  # node 2 is problematic
				logs.write(f"\t=> Removing node {node2}\n")
				nodes_to_remove.append(node2)
			else:  # Couldn't decide which one is problematic, Remove the connection between them.
				logs.write(f"\t=> Removing the connection\n")
				edges_to_remove.append((node1, node2))

		logs.write("\n\nComplete list:\n")
		for node in set(nodes_to_remove):
			logs.write(f"\t- Node {node} - {graph.nodes[node]}\n")
			graph.remove_node(node)
		for edge in edges_to_remove:
			if graph.has_edge(*edge):
				logs.write(f"\t- Edge {edge}\n")
				graph.remove_edge(*edge)

				for node in edge:  # If a node is left alone, remove it.
					if len(graph.edges(node)) == 0:
						logs.write(f"\t\t(also removed node {node} - {graph.nodes[node]} because left alone)\n")
						graph.remove_node(node)

		logs.close()

	@staticmethod
	def separate_communities(graph: nx.Graph) -> None:
		"""
		Looks for all subgraphs (connected component) and uses the Louvain algorithm to check if
		multiple communities are found. If so, the edges between communities are removed.
		Additionally, small communities are removed.
		Warning: it's recommended to run this function if there are at least 4 analyses.

		@param graph: nx.Graph
			The graph containing all the units and connected by their similarity.
		"""

		for nodes in list(nx.connected_components(graph)):
			subgraph = graph.subgraph(nodes)
			communities = list(nx.community.louvain_communities(subgraph, resolution=0.85))

			if len(communities) == 1:
				continue

			# Remove edges between communities
			for i in range(len(communities) - 1):
				for j in range(i+1, len(communities)):
					for node1 in communities[i]:
						for node2 in communities[j]:
							if graph.has_edge(node1, node2):
								graph.remove_edge(node1, node2)

			# Remove small communities
			for community in communities:
				if len(community) <= 2:
					for node in community:
						graph.remove_node(node)

	def merge_sortings(self, graph: nx.Graph, params: dict) -> si.NpzSortingExtractor:
		"""
		Merges the sortings based on a graph of similar units.

		@param graph: nx.Graph
			The graph containing all the units and connected by their similarity.
		@param params: dict
			The parameters of the merge_sorting module.
		@return merged_sorting: si.NpzSortingExtractor
			The merged sorting.
		"""

		k = 2.5
		t_c = int(round(params['refractory_period'][0] * 1e-3 * self.recording.sampling_frequency))
		new_spike_trains = {}
		logs = open(f"{self.logs_folder}/merge_sortings_logs.txt", 'w+')

		for nodes in nx.connected_components(graph):  # For each putative neuron.
			nodes = list(nodes)

			new_unit_id = len(new_spike_trains)
			best_score = -100000
			unit_label = ""
			logs.write(f"\nMaking unit {new_unit_id} from {nodes}\n")

			if len(nodes) == 1 and params['require_multiple_sortings_match']:  # Be more strict about nodes that end up being alone.
				node = nodes[0]
				if graph.nodes[node]['contamination'] > 0.10:
					logs.write(f"\t- Contamination too high (C = {graph.nodes[node]['contamination']:.1%})\n\t--> SKIPPING\n")
					continue
				if graph.nodes[node]['SNR'] < 3.0:
					logs.write(f"\t- SNR too low (SNR = {graph.nodes[node]['SNR']:.2f})\n\t--> SKIPPING\n")
					continue
				if abs(graph.nodes[node]['sd_ratio'] - 1) > 0.2:
					logs.write(f"\t- SD ratio too different from 1.0 (SD ratio = {graph.nodes[node]['sd_ratio']:.2f})\n\t--> SKIPPING\n")
					continue

			for n_units in range(1, params['max_units_merge']+1):
				for sub_nodes in itertools.combinations(nodes, n_units):
					spike_trains = [self.sortings[name].get_unit_spike_train(unit_id) for name, unit_id in sub_nodes]
					spike_train = np.sort(list(itertools.chain(*spike_trains))).astype(np.int64)
					indices_of_duplicates = find_duplicated_spikes(spike_train, t_c, method="keep_first_iterative")
					spike_train = np.delete(spike_train, indices_of_duplicates)

					f = len(spike_train) * self.sampling_f / self.recording.get_num_frames()
					C = utils.estimate_contamination(spike_train, params['refractory_period'])
					score = f * (1 - (k+1)*C)
					logs.write(f"\t- Score = {score:.2f}\t[{sub_nodes}]\n")

					if score > best_score:
						best_score = score
						new_spike_trains[new_unit_id] = spike_train
						unit_label = sub_nodes

			logs.write(f"\t--> Unit(s) chosen: {unit_label}\n")

		logs.close()

		sorting = si.NumpySorting.from_unit_dict(new_spike_trains, self.sampling_f)
		filename = f"{self.logs_folder}/merged_sorting.npz"
		si.NpzSortingExtractor.write_sorting(sorting, filename)

		merged_sorting = si.NpzSortingExtractor(filename)
		merged_sorting.annotate(name="merged_sorting")

		return merged_sorting
