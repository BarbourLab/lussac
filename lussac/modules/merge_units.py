from typing import Any
import numpy as np
from overrides import override
import plotly.graph_objects as go
from lussac.core.module import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur
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
				'max_spikes_per_unit': 2000,
				'filter': {
					'band': [100, 9000],
					'filter_order': 2,
					'ftype': 'bessel'
				}
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
		wvf_extractor = self.extract_waveforms(**params['wvf_extraction'])
		potential_merges, extra_outputs = scur.get_potential_auto_merge(wvf_extractor, extra_outputs=True, **params['auto_merge_params'])

		self.plot_merging(potential_merges, wvf_extractor, extra_outputs, params['auto_merge_params'])

		sorting = scur.CurationSorting(self.sorting)
		# TODO

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

		self.plot_results(extra_outputs, params, wvf_extractor)
		self.plot_difference_matrix(extra_outputs, params)

	def plot_results(self, extra_outputs: dict[str, Any], params: dict[str, Any], wvf_extractor: si.WaveformExtractor) -> None:
		"""
		TODO

		@param extra_outputs:
		@param params:
		@param wvf_extractor:
		"""
		bins = extra_outputs['bins']
		correlograms = extra_outputs['correlograms']
		correlograms_smoothed = extra_outputs['correlograms_smoothed']
		correlogram_diff = extra_outputs['correlogram_diff']
		templates_diff = extra_outputs['templates_diff']
		window_sizes = extra_outputs['win_sizes']
		corr_diff_threshold = params['corr_diff_thresh']
		template_diff_threshold = params['template_diff_thresh']

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
			labels.append(f"Units {unit_id_1} & {unit_id_2}")
			args.append({'title.text': f"Units {unit_id_1} & {unit_id_2}: corr_diff = {correlogram_diff[i, j]:.1%} ; temp_diff = {templates_diff[i, j]:.1%}",
						 'title.font.color': "black" if correlogram_diff[i, j] <= corr_diff_threshold else "red"})  # color based on if in potential_merges.

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
		fig.update_yaxes(title_text="Templates difference")

		utils.plotting.export_figure(fig, f"{self.logs_folder}/difference_matrix.html")

