from typing import Any
import networkx as nx
import numpy as np
from overrides import override
import plotly.graph_objects as go
from lussac.core import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur
import spikeinterface.qualitymetrics as sqm
from spikeinterface.curation.auto_merge import normalize_correlogram


class MergeUnits(MonoSortingModule):
	"""
	Detects and merges units that come from the same neuron if there is an increase in score.
	If there is a decrease in score, the worse unit is removed.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'refractory_period': [0.2, 1.0],
			'wvf_extraction': {
				'ms_before': 1.0,
				'ms_after': 1.5,
				'max_spikes_per_unit': 2_000,
				'filter': [100, 9000]
			},
			'auto_merge_params': {
				'bin_ms': 0.05,
				'window_ms': 150,
				'corr_diff_thresh': 0.16,
				'template_diff_thresh': 0.25,
				'firing_contamination_balance': 2.5
			}
		}

	@override
	def update_params(self, params: dict[str, Any]) -> dict[str, Any]:
		params = super().update_params(params)

		params['auto_merge_params']['censored_period_ms'] = params['refractory_period'][0]
		params['auto_merge_params']['refractory_period_ms'] = params['refractory_period'][1]

		return params

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		wvf_extractor = self.extract_waveforms(sparse=False, **params['wvf_extraction'])
		potential_merges, extra_outputs = scur.get_potential_auto_merge(wvf_extractor, extra_outputs=True, **params['auto_merge_params'])

		sorting = self._remove_splits(self.sorting, extra_outputs, params)
		sorting = self._merge(sorting, potential_merges, params)
		self.plot_merging(potential_merges, wvf_extractor, extra_outputs, params['auto_merge_params'])

		return sorting

	def _remove_splits(self, sorting: si.BaseSorting, extra_outputs: dict, params: dict[str, Any]) -> si.BaseSorting:
		"""
		Remove units that are split but decrease the score if merged.
		When such a pair is detected, remove the unit with the lowest score.
		This unit usually is a contaminated unit.
		TODO: Create a plot of removed units (logs).

		@param sorting: si.BaseSorting
			The sorting object.
		@param extra_outputs: dict
			Extra outputs given by the merging process.
		@param params: dict
			Parameters of the merging process.
			i.e. params['auto_merge_params']
		@return: si.BaseSorting
			Sorting with the units removed.
		"""
		t_c, t_r = params['refractory_period']
		k = params['auto_merge_params']['firing_contamination_balance']

		units_to_remove = []
		wvf_extractor = si.WaveformExtractor(self.recording, sorting, allow_unfiltered=True)
		contamination, _ = sqm.compute_refrac_period_violations(wvf_extractor, refractory_period_ms=t_r, censored_period_ms=t_c)

		for pair in extra_outputs['pairs_decreased_score']:
			unit1, unit2 = pair
			score_unit1 = len(sorting.get_unit_spike_train(unit1)) * (1 - (k+1) * contamination[unit1])
			score_unit2 = len(sorting.get_unit_spike_train(unit2)) * (1 - (k+1) * contamination[unit2])
			worse_unit = unit1 if score_unit1 < score_unit2 else unit2
			units_to_remove.append(worse_unit)  # TODO: Only remove if unconnected in 'potential_merges' graph?

		return sorting.select_units([unit_id for unit_id in sorting.unit_ids if unit_id not in units_to_remove])

	def _merge(self, sorting: si.BaseSorting, potential_merges: list[tuple], params: dict[str, Any]) -> si.BaseSorting:
		"""
		Merges units based on a list of potential merges.
		Creates a graph with all the units, where potential merges are the edges.
		For each connected component (i.e. each putative neuron), go through all the edges and compute the
		score of the merge. If the score increases (over lone unit scores), merges the best edge.
		This is done iteratively until either one unit remains or no increase is detected.
		The unit not merged (because of decrease in score) are deleted.

		@param sorting: si.BaseSorting
			The sorting object.
		@param potential_merges: list[tuple]
			List of potential merges.
		@param params: dict
			Parameters of the merging process.
			i.e. params['auto_merge_params']
		@return: si.BaseSorting
			Sorting with the splits merged and removed.
		"""
		t_c, t_r = params['refractory_period']
		k = params['auto_merge_params']['firing_contamination_balance']

		wvf_extractor = si.WaveformExtractor(self.recording, sorting, allow_unfiltered=True)
		contamination, _ = sqm.compute_refrac_period_violations(wvf_extractor, refractory_period_ms=t_r, censored_period_ms=t_c)
		sorting = scur.CurationSorting(sorting, properties_policy="keep")

		graph = nx.Graph()
		for potential_merge in potential_merges:
			unit1, unit2 = potential_merge
			if unit1 not in sorting.sorting.unit_ids or unit2 not in sorting.sorting.unit_ids:
				continue  # Can happen if a unit is removed in '_remove_splits'.

			for unit in [unit1, unit2]:
				if unit not in graph:
					score = len(sorting.sorting.get_unit_spike_train(unit)) * (1 - (k+1) * contamination[unit])
					graph.add_node(unit, score=score)

			graph.add_edge(*potential_merge)

		# For each putative neuron, merge the units.
		for units in nx.connected_components(graph):
			subgraph = graph.subgraph(units).copy()

			while len(subgraph) > 1:
				highest_score = max(dict(subgraph.nodes(data="score")).values())
				merge = None

				for unit1, unit2 in subgraph.edges:
					sorting_merged = scur.MergeUnitsSorting(sorting.sorting, [[unit1, unit2]], new_unit_ids=[unit1], delta_time_ms=t_c).select_units([unit1])
					wvf_extractor = si.WaveformExtractor(self.recording, sorting_merged, allow_unfiltered=True)
					C = sqm.compute_refrac_period_violations(wvf_extractor, refractory_period_ms=t_r, censored_period_ms=t_c)[0][unit1]
					score = len(sorting_merged.get_unit_spike_train(unit1)) * (1 - (k+1) * C)

					if score > highest_score:
						highest_score = score
						merge = (unit1, unit2)

				if merge is None:
					scores = dict(subgraph.nodes(data="score"))
					best_unit = max(scores, key=scores.get)
					sorting.remove_units([x for x in subgraph.nodes if x != best_unit])
					break

				unit1, unit2 = merge
				sorting.merge(merge, new_unit_id=unit1)
				subgraph = nx.contracted_nodes(subgraph, unit1, unit2, self_loops=False)
				subgraph.nodes[unit1]['score'] = highest_score

		return sorting.sorting

	def plot_merging(self, potential_merges: list[tuple], wvf_extractor: si.WaveformExtractor, extra_outputs: dict[str, Any], params: dict[str, Any]) -> None:
		"""
		Makes different plots about the merging process.

		@param potential_merges: list[tuple]
			List of potential merges pairwise.
		@param wvf_extractor: WaveformExtractor
			Waveform extractor used to extract the waveforms.
		@param extra_outputs: dict[str, Any]
			Extra outputs given by the merging process.
		@param params: dict[str, Any]
			Parameters given to the merging process.
			i.e. params['auto_merge_params']
		"""

		self.plot_results(potential_merges, extra_outputs, params, wvf_extractor)
		self.plot_difference_matrix(extra_outputs, params)

	def plot_results(self, potential_merges: list[tuple], extra_outputs: dict[str, Any], params: dict[str, Any], wvf_extractor: si.WaveformExtractor) -> None:
		"""
		Plots the result of the merging process (i.e. the pairs merged or closed to be merged).

		@param potential_merges: list[tuple]
			List of potential merges pairwise (containing the unit ids).
		@param extra_outputs: dict[str, Any]
			The extra outputs given by the merging process.
		@param params: dict[str, Any]
			The parameters given to the merging process.
		@param wvf_extractor: WaveformExtractor
			The waveform extractor used by the merging process.
		"""
		bins = extra_outputs['bins']
		correlograms = extra_outputs['correlograms']
		correlograms_smoothed = extra_outputs['correlograms_smoothed']
		correlogram_diff = extra_outputs['correlogram_diff']
		templates_diff = extra_outputs['templates_diff']
		window_sizes = extra_outputs['win_sizes']
		corr_diff_threshold = params['corr_diff_thresh']

		fig = go.Figure().set_subplots(rows=2, cols=2)
		bins = bins[:-1] + (bins[1] - bins[0]) / 2
		t_axis = np.arange(-wvf_extractor.nbefore, wvf_extractor.nafter) / wvf_extractor.sampling_frequency * 1e3
		wvfs_unit = "µV" if wvf_extractor.return_scaled else "A.U."
		labels = []
		args = []

		for k in range(correlograms.shape[0]**2):
			i, j = k // correlograms.shape[0], k % correlograms.shape[0]
			if i == j:
				continue
			if correlogram_diff[i, j] > corr_diff_threshold + 0.05 or np.isnan(correlogram_diff[i, j]):
				continue

			unit_id_1 = self.sorting.unit_ids[i]
			unit_id_2 = self.sorting.unit_ids[j]
			color = "black" if (unit_id_1, unit_id_2) in potential_merges or (unit_id_2, unit_id_1) in potential_merges else "red"

			annotation_gt = {}
			if 'gt_label' in wvf_extractor.sorting.get_property_keys():
				gt_label_1 = wvf_extractor.sorting.get_unit_property(unit_id_1, 'gt_label')
				gt_label_2 = wvf_extractor.sorting.get_unit_property(unit_id_2, 'gt_label')
				annotation_gt = {
					'x': 1.0,
					'y': 1.05,
					'xref': "paper",
					'yref': 'paper',
					'xanchor': 'right',
					'yanchor': 'top',
					'text': f"GT {unit_id_1}: {gt_label_1}<br />GT {unit_id_2}: {gt_label_2}",
					'font': {
						'color': "rgba(73, 42, 189, 1.0)"
					},
					'showarrow': False
				}

			labels.append(f"Units {unit_id_1} & {unit_id_2}")
			args.append({'title.text': f"Units {unit_id_1} & {unit_id_2}: corr_diff = {correlogram_diff[i, j]:.1%} ; temp_diff = {templates_diff[i, j]:.1%}",
						 'title.font.color': color,
						 'annotations': [annotation_gt]})

			fig.add_trace(go.Scatter(
				x=bins,
				y=correlograms[i, i, :],
				mode="lines",
				name=f"Auto-corr unit {unit_id_1}",
				marker_color="CornflowerBlue"
			), row=1, col=1)
			fig.add_trace(go.Scatter(
				x=bins,
				y=correlograms[j, j, :],
				mode="lines",
				name=f"Auto-corr unit {unit_id_2}",
				marker_color="LightSeaGreen"
			), row=1, col=1)
			fig.add_trace(go.Scatter(
				x=bins,
				y=correlograms[i, j, :],
				mode="lines",
				name=f"Cross-corr unit {unit_id_1} - {unit_id_2}",
				marker_color="Crimson"
			), row=1, col=1)

			fig.add_trace(go.Scatter(
				x=bins,
				y=normalize_correlogram(correlograms_smoothed[i, i, :]),
				mode="lines",
				name=f"Auto-corr unit {unit_id_1}",
				marker_color="CornflowerBlue"
			), row=1, col=2)
			fig.add_trace(go.Scatter(
				x=bins,
				y=normalize_correlogram(correlograms_smoothed[j, j, :]),
				mode="lines",
				name=f"Auto-corr unit {unit_id_2}",
				marker_color="LightSeaGreen"
			), row=1, col=2)
			fig.add_trace(go.Scatter(
				x=bins,
				y=normalize_correlogram(correlograms_smoothed[i, j, :]),
				mode="lines",
				name=f"Cross-corr unit {unit_id_1} - {unit_id_2}",
				marker_color="Crimson"
			), row=1, col=2)
			# TODO: Plot window size for correlogram.

			templates1 = wvf_extractor.get_template(unit_id_1)
			templates2 = wvf_extractor.get_template(unit_id_2)
			best_channels = np.argsort(np.max(np.abs(templates1) + np.abs(templates2), axis=0))[::-1]
			for i in range(2):
				channel_id = wvf_extractor.recording.get_channel_ids()[best_channels[i]]
				fig.add_trace(go.Scatter(
					x=t_axis,
					y=templates1[:, best_channels[i]],
					mode="lines",
					name=f"Template unit {unit_id_1} (channel {channel_id})",
					marker_color="CornflowerBlue"
				), row=2, col=1+i)
				fig.add_trace(go.Scatter(
					x=t_axis,
					y=templates2[:, best_channels[i]],
					mode="lines",
					name=f"Template unit {unit_id_2} (channel {channel_id})",
					marker_color="LightSeaGreen"
				), row=2, col=1+i)

		fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
		fig.update_xaxes(title_text="Time (ms)", matches='x', row=1, col=2)
		fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
		fig.update_xaxes(title_text="Time (ms)", matches='x3', row=2, col=2)
		fig.update_yaxes(title_text=f"Voltage ({wvfs_unit})", row=2, col=1)
		fig.update_yaxes(title_text=f"Voltage ({wvfs_unit})", matches='y3', row=2, col=2)

		utils.plotting.plot_sliders(fig, 10, labels, f"{self.logs_folder}/results", args=args)

	def plot_difference_matrix(self, extra_outputs: dict[str, Any], params: dict[str, Any]) -> None:
		"""
		Plots the difference matrix between units pairwise.
		i.e. for each pair, plots the correlogram (x-axis) and template (y-axis) difference.

		@param extra_outputs: dict[str, Any]
			Extra outputs given by the merging process.
		@param params: dict[str, Any]
			Parameters given to the merging process.
			i.e. params['auto_merge_params']
		"""
		correlogram_diff = extra_outputs['correlogram_diff']
		templates_diff = extra_outputs['templates_diff']
		np.save(f"{self.logs_folder}/difference_matrix.npy", np.dstack((correlogram_diff, templates_diff)))

		unit_ids = self.sorting.unit_ids
		N = len(unit_ids)
		text = [f"Units {unit_ids[i//N]} - {unit_ids[i%N]}" for i in range(len(correlogram_diff)**2)]

		fig = go.Figure()

		fig.add_trace(go.Scatter(
			x=correlogram_diff.flatten(),
			y=templates_diff.flatten(),
			mode="markers",
			text=text,
			marker_color="CornflowerBlue"
		))

		fig.add_shape(
			type="rect",
			x0=0, x1=params['corr_diff_thresh'],
			y0=0, y1=params['template_diff_thresh'],
			line={'color': "Crimson", 'dash': "dash"}
		)

		fig.update_xaxes(title_text="Correlograms difference")
		fig.update_yaxes(title_text="Templates difference", range=[-0.03, 1.03])

		utils.plotting.export_figure(fig, f"{self.logs_folder}/difference_matrix.html")
