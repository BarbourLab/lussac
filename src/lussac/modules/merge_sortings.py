import copy
import itertools
import math
import pickle
import shutil
from typing import Any
from overrides import override
import networkx as nx
import numpy as np
import scipy.stats
from lussac.core import MultiSortingsModule, TemplateExtractor
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur
from spikeinterface.curation.curation_tools import find_duplicated_spikes
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm


class MergeSortings(MultiSortingsModule):
	"""
	Merges the sortings into a single one.
	"""

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
				'max_difference': 0.25,
				'gaussian_std': 0.6,
				'gaussian_truncate': 5.0
			},
			'waveform_validation': {
				'max_difference': 0.20,
				'wvf_extraction': {
					'ms_before': 1.0,
					'ms_after': 2.0,
					'max_spikes_per_unit': 1_000
				},
				'filter': [200.0, 6_000.0],
				'margin_ms': 5.0,
				'num_channels': 5
			},
			'merge_check': {
				'cross_cont_limit': 0.22
			}
		}

	@override
	def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
		params = super().update_params(params)

		assert params['max_shift'] < params['waveform_validation']['margin_ms']

		params['max_shift'] = int(round(params['max_shift'] * 1e-3 * self.sampling_f))
		params['similarity']['window'] = int(round(params['similarity']['window'] * 1e-3 * self.sampling_f))
		params['correlogram_validation']['max_time'] = int(round(params['correlogram_validation']['max_time'] * 1e-3 * self.sampling_f))
		params['correlogram_validation']['gaussian_std'] = params['correlogram_validation']['gaussian_std'] * 1e-3 * self.sampling_f
		params['correlogram_validation']['censored_period'] = params['similarity']['window']
		params['waveform_validation']['margin_ms'] = max(1.0, params['waveform_validation']['margin_ms'])

		return params

	@override
	def run(self, params: dict[str, Any]) -> dict[str, si.BaseSorting]:
		cross_shifts = self.compute_cross_shifts(params['max_shift'])

		similarity_matrices = self._compute_similarity_matrices(cross_shifts, params)
		graph = self._compute_graph(similarity_matrices, params)
		self.compute_correlogram_difference(graph, cross_shifts, params['correlogram_validation'])
		self.compute_waveform_difference(graph, cross_shifts, params['waveform_validation'])

		if params['merge_check']:
			self.remove_merged_units(graph, cross_shifts, params['refractory_period'], params['merge_check'])
		self.clean_graph(graph, params)
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
		cross_shifts = {name1: {name2: 0 for name2 in self.sortings.keys() if name1 != name2} for name1 in self.sortings.keys()}

		for i, (name1, sorting1) in enumerate(self.sortings.items()):
			for j, (name2, sorting2) in enumerate(self.sortings.items()):
				if i >= j:
					continue

				shifts = utils.compute_cross_shift_from_vector(spike_vectors[name1], spike_vectors[name2], max_shift)

				cross_shifts[name1][name2] = shifts
				cross_shifts[name2][name1] = -shifts.T

		return cross_shifts

	def _compute_similarity_matrices(self, cross_shifts: dict[str, dict[str, np.ndarray]], params: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
		"""
		Computes the similarity matrix between all sortings.

		@param cross_shifts: dict[str, dict[str, np.ndarray]]
			The cross-shifts between units.
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
		min_f, max_f = params['waveform_validation']['filter']
		recording_f = spre.gaussian_bandpass_filter(self.recording, freq_min=min_f, freq_max=max_f)

		# Populating the graph with all the nodes (i.e. all the units) with properties.
		graph = nx.Graph()
		for (name, sorting) in self.sortings.items():
			if sorting.get_num_units() == 0: continue
			wvf_extractor = si.extract_waveforms(recording_f, sorting, folder=self.tmp_folder / f"wvfs_{name}", ms_before=1.5, ms_after=1.5, max_spikes_per_unit=150, sparse=False)
			spost.compute_spike_amplitudes(wvf_extractor, peak_sign="both", return_scaled=recording_f.has_scaled())
			contamination, _ = sqm.compute_refrac_period_violations(wvf_extractor, refractory_period_ms=refractory_period, censored_period_ms=censored_period)
			# sd_ratio = sqm.compute_sd_ratio(wvf_extractor)

			for unit_id in sorting.unit_ids:
				attr = {key: sorting.get_unit_property(unit_id, key) for key in sorting.get_property_keys() if key.startswith('gt_')}
				attr['contamination'] = contamination[unit_id]
				# attr['sd_ratio'] = sd_ratio[unit_id]
				graph.add_node((name, unit_id), **attr)

			del wvf_extractor
			shutil.rmtree(self.tmp_folder / f"wvfs_{name}")

		# Connecting nodes using the similarity.
		for i, (name1, sorting1) in enumerate(self.sortings.items()):
			for j, (name2, sorting2) in enumerate(self.sortings.items()):
				if j <= i:
					continue

				for unit_ind1, unit_id1 in enumerate(sorting1.unit_ids):
					for unit_ind2, unit_id2 in enumerate(sorting2.unit_ids):
						if (similarity := similarity_matrices[name1][name2][unit_ind1, unit_ind2]) >= params['similarity']['min_similarity']:
							graph.add_edge((name1, unit_id1), (name2, unit_id2), similarity=similarity, problem=False)

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

				spike_train1 = self.sortings[sorting1_name].get_unit_spike_train(unit_id1) + cross_shifts[sorting_name][sorting1_name][unit_ind, unit_ind1]
				spike_train2 = self.sortings[sorting2_name].get_unit_spike_train(unit_id2) + cross_shifts[sorting_name][sorting2_name][unit_ind, unit_ind2]
				C1 = graph.nodes[(sorting1_name, unit_id1)]['contamination']
				C2 = graph.nodes[(sorting2_name, unit_id2)]['contamination']
				if C2 < C1:
					spike_train1, spike_train2 = spike_train2, spike_train1
				cross_cont, p_value = utils.estimate_cross_contamination(spike_train1, spike_train2, refractory_period, limit=params['cross_cont_limit'])

				logs.write(f"\nUnit {node} is connected to {node1} and {node2}:\n")
				logs.write(f"\tcross-cont = {cross_cont:.2%} (p_value={p_value:.3f})\n")
				logs.write(f"\tC1 = {C1:.1%} ; C2 = {C2:.1%}\n")
				if p_value > 5e-3:  # No problem, node1 and node2 are probably just a split.
					continue

				spike_train = self.sortings[sorting_name].get_unit_spike_train(unit_id)
				C = graph.nodes[(sorting_name, unit_id)]['contamination']
				cross_cont1, p_value1 = utils.estimate_cross_contamination(spike_train1, spike_train, refractory_period, limit=0.1)
				cross_cont2, p_value2 = utils.estimate_cross_contamination(spike_train2, spike_train, refractory_period, limit=0.1)
				p_value1, p_value2 = 1 - p_value1, 1 - p_value2  # Reverse the p-values because we want to know the probability <= and not >=.

				logs.write(f"\tcheck1 = {cross_cont1:.2%} (p_value={p_value1:.3f})\n")
				logs.write(f"\tcheck2 = {cross_cont2:.2%} (p_value={p_value2:.3f})\n")

				if p_value1 < 1e-3:  # node2 is the problematic unit.
					if node2 not in nodes_to_remove and C2 > C + 0.02:
						nodes_to_remove.append(node2)
						logs.write(f"\t-> Unit {node2} is considered a problematic unit.\n")
					continue
				elif p_value2 < 1e-3:  # node1 is the problematic unit.
					if node1 not in nodes_to_remove and C1 > C + 0.02:
						nodes_to_remove.append(node1)
						logs.write(f"\t-> Unit {node1} is considered a problematic unit.\n")
					continue
				elif min(C1, C2) < 0.1 and C > min(C1, C2) + 0.02:  # node is probably a merged unit.
					if node not in nodes_to_remove:
						nodes_to_remove.append(node)
						logs.write(f"\t-> Unit {node} is considered a merged unit.\n")

		if len(nodes_to_remove) > 0:
			logs.write("\nRemoved units:\n")
			for node in nodes_to_remove:  # Remove node then re-add it no remove all the edges.
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
		gaussian = scipy.stats.norm.pdf(np.arange(-N, N+1), loc=0.0, scale=params['gaussian_std'])

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
			The parameters for the waveform difference.
		"""

		params = copy.deepcopy(params)
		n_channels = params['num_channels']
		margin = round(params['margin_ms'] * self.sampling_f * 1e-3)
		params['wvf_extraction']['ms_before'] += params['margin_ms']
		params['wvf_extraction']['ms_after'] += params['margin_ms']
		params_channels = {
			'ms_before': min(params['wvf_extraction']['ms_before'], 2.0),
			'ms_after': min(params['wvf_extraction']['ms_after'], 2.0)
		}

		template_extractors = {name: TemplateExtractor(self.recording, sorting, self.tmp_folder, params['wvf_extraction']) for name, sorting in self.sortings.items()}

		for node1, node2 in graph.edges:
			sorting1_name, unit_id1 = node1
			sorting2_name, unit_id2 = node2
			unit_idx1 = self.sortings[sorting1_name].id_to_index(unit_id1)
			unit_idx2 = self.sortings[sorting2_name].id_to_index(unit_id2)

			best_channels1 = template_extractors[sorting1_name].get_unit_best_channels(unit_id1, **params_channels)
			best_channels2 = template_extractors[sorting2_name].get_unit_best_channels(unit_id2, **params_channels)
			channel_ids = np.unique(np.concatenate((best_channels1[:n_channels], best_channels2[:n_channels])))

			template1 = template_extractors[sorting1_name].get_template(unit_id1, channel_ids, return_scaled=self.recording.has_scaled())
			template2 = template_extractors[sorting2_name].get_template(unit_id2, channel_ids, return_scaled=self.recording.has_scaled())
			template1 = utils.filter(template1, params['filter'], axis=0)
			template2 = utils.filter(template2, params['filter'], axis=0)
			channel_indices = np.argsort(np.max(np.abs(template1) + np.abs(template2), axis=0))[:-n_channels-1:-1]

			shift = cross_shifts[sorting1_name][sorting2_name][unit_idx1, unit_idx2]
			template1 = template1[margin:-margin, channel_indices]
			template2 = template2[margin+shift:shift-margin, channel_indices]

			template_diff = np.sum(np.abs(template1 - template2)) / np.sum(np.abs(template1) + np.abs(template2))
			graph[node1][node2]['temp_diff'] = template_diff

	def clean_graph(self, graph: nx.Graph, params: dict) -> None:  # pragma: no cover (not implemented yet)
		"""
		TODO

		@param graph:
		@param params:
		@return:
		"""

		nodes_to_remove = []

		for node1, node2, data in list(graph.edges(data=True)):
			if node1 in nodes_to_remove or node2 in nodes_to_remove:
				continue

			if 'corr_diff' not in data and 'temp_diff' not in data:
				continue

			if ('corr_diff' in data and data['corr_diff'] > 0.25) or ('temp_diff' in data and data['temp_diff'] > 0.20):
				data['problem'] = True
				sorting1_name, unit_id1 = node1
				sorting2_name, unit_id2 = node2

				C1 = utils.estimate_contamination(self.sortings[sorting1_name].get_unit_spike_train(unit_id1), refractory_period=params['refractory_period'])
				C2 = utils.estimate_contamination(self.sortings[sorting2_name].get_unit_spike_train(unit_id2), refractory_period=params['refractory_period'])

				"""if C1 > C2 + 0.04:
					nodes_to_remove.append(node1)
				elif C2 > C1 + 0.04:
					nodes_to_remove.append(node2)"""

		for node in nodes_to_remove:
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
