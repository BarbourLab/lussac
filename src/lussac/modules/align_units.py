from typing import Any
from overrides import override
import numpy as np
import plotly.graph_objects as go
import scipy.signal
from lussac.core import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si
import spikeinterface.curation as scur
import spikeinterface.postprocessing as spost


class AlignUnits(MonoSortingModule):
	"""
	Align the units so that their peak is at t=0 in the spike train.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'wvf_extraction': {
				'ms_before': 2.0,
				'ms_after': 2.0,
				'max_spikes_per_unit': 1_000
			},
			'filter': [200, 5000],
			'threshold': 0.5,
			'check_next': 10
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		templates, wvf_extractor, margin = self.get_templates(params['wvf_extraction'], params['filter'], return_extractor=True)
		best_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
		templates = templates[np.arange(templates.shape[0]), :, best_channels]  # Only take the best channel for each unit.

		shifts = self.get_units_shift(templates, wvf_extractor.nbefore - margin, params['threshold'], params['check_next'])
		self._plot_alignment(templates, wvf_extractor.nbefore - margin, shifts, params['threshold'])

		return self.shift_sorting(self.recording, self.sorting, {self.sorting.unit_ids[i]: -shifts[i] for i in range(len(shifts))})

	@staticmethod
	def shift_sorting(recording: si.BaseRecording, sorting: si.BaseSorting, shift: dict[Any, int]) -> si.BaseSorting:
		"""
		Shifts the spike train of the sorting by shift time samples.

		@param recording: si.BaseRecording
			The recording object (needed to check for spike going over the edges).
		@param sorting: si.BaseSorting
			The sorting to shift.
		@param shift: dict[Any, int]
			A dict mapping the unit id to its shift.
		@return aligned_sorting: si.BaseSorting
			The shifted sorting.
		"""

		aligned_sorting = spost.align_sorting(sorting, shift)
		aligned_sorting = scur.remove_excess_spikes(aligned_sorting, recording)

		return aligned_sorting

	@staticmethod
	def get_units_shift(templates: np.ndarray, nbefore: int, threshold: float, check_next: int) -> np.ndarray:
		"""
		Computes the shift between nbefore and the peak of the template.

		@param templates: np.ndarray (n_units, n_samples)
			The template for each unit.
		@param nbefore: int
			The index of t=0 in the templates.
		@param threshold: float
			First the first peak that is at least threshold*max.
		@param check_next: int
			Checks the next x time samples after the first peak, and take the highest one.
		@return units_shift: np.ndarray[int16] (n_units)
			The shift
		"""

		centers = np.empty(templates.shape[0], dtype=np.int16)

		for i, template in enumerate(templates):
			template = np.abs(template)
			peaks, _ = scipy.signal.find_peaks(template, height=threshold * np.max(template))

			if len(peaks) == 0:  # Can happen if the maximum is at the very start of end of the template.
				peaks = [np.argmax(template)]

			centers[i] = peaks[0] + np.argmax(template[peaks[0]:peaks[0] + check_next + 1])

		return centers - nbefore

	def _plot_alignment(self, templates: np.ndarray, nbefore: int, shifts: np.ndarray, threshold_ratio: float) -> None:
		"""
		Plots the template for each unit with the new center.

		@param templates: np.ndarray (n_units, n_samples)
			The template for each unit on its best channel.
		@param nbefore: int
			The index of t=0 in the templates.
		@param shifts: np.ndarray (n_units)
			The new center for each unit.
		@param threshold_ratio: float
			The threshold parameter used to compute the shifts.
		"""

		xaxis = (np.arange(templates.shape[1]) - nbefore) / utils.Utils.sampling_frequency * 1e3
		fig = go.Figure()
		labels = []
		args = []

		old_center_line = {'x0': 0, 'x1': 0, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'y domain', 'line': {'dash': "dash", 'color': "GoldenRod"}}  #, 'annotation': {'text': "Old Center", 'textangle': -90}}
		fig.add_shape(**old_center_line)
		fig.update_xaxes(title_text="Time relative to old center (ms)")
		
		for i in range(templates.shape[0]):
			unit_id = self.sorting.unit_ids[i]
			shift = shifts[i] / utils.Utils.sampling_frequency * 1e3
			threshold = threshold_ratio * np.max(np.abs(templates[i]))

			fig.add_trace(go.Scatter(
				x=xaxis,
				y=templates[i],
				mode="lines",
				marker_color="CornflowerBlue"
			))

			new_center_line = {'x0': shift, 'x1': shift, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'y domain', 'line': {'dash': "dash", 'color': "Crimson"}}  #, 'annotation': {'text': "New Center", 'textangle': -90}}
			threshold_line_up = {'x0': 0, 'x1': 1, 'y0': threshold, 'y1': threshold, 'xref': 'x domain', 'yref': 'y', 'line': {'dash': "dash", 'color': "LightSeaGreen"}, 'opacity': 0.4}
			threshold_line_down = {'x0': 0, 'x1': 1, 'y0': -threshold, 'y1': -threshold, 'xref': 'x domain', 'yref': 'y', 'line': {'dash': "dash", 'color': "LightSeaGreen"}, 'opacity': 0.4}
			if i == 0:
				fig.add_shape(**new_center_line)
				fig.add_shape(**threshold_line_up)
				fig.add_shape(**threshold_line_down)

			labels.append(f"Unit {unit_id}")
			args.append({
				'title_text': f"Unit {unit_id}",
				'shapes': [old_center_line, new_center_line, threshold_line_up, threshold_line_down]
			})

		utils.plot_sliders(fig, 1, labels, f"{self.logs_folder}/alignment", args)
