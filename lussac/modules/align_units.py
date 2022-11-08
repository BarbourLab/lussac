from typing import Any
from overrides import override
import numpy as np
import plotly.graph_objects as go
import scipy.signal
from lussac.core.module import MonoSortingModule
import lussac.utils as utils
import spikeinterface.core as si
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
				'max_spikes_per_unit': 2000
			},
			'filter': [200, 5000],
			'threshold': 0.5
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		templates, wvf_extractor, margin = self.get_templates(params['wvf_extraction'], params['filter'], return_extractor=True)
		best_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
		templates = templates[np.arange(templates.shape[0]), :, best_channels]  # Only take the best channel for each unit.

		# Wrong nbefore because of margins.
		shifts = self.get_units_shift(templates, wvf_extractor.nbefore - margin, params['threshold'])
		self._plot_alignment(templates, wvf_extractor.nbefore - margin, shifts, params['threshold'])

		return spost.align_sorting(self.sorting, shifts)

	@staticmethod
	def get_units_shift(templates: np.ndarray, nbefore: int, threshold: float, check_next: int = 10) -> np.ndarray:
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

			centers[i] = peaks[0] + np.argmax(template[peaks[0]:peaks[0] + check_next])

		return centers - nbefore

	def _plot_alignment(self, templates: np.ndarray, nbefore: int, shifts: np.ndarray, threshold: float) -> None:
		"""
		TODO

		@param templates:
		@param nbefore:
		@param shifts:
		@param threshold:
		@return:
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

			fig.add_trace(go.Scatter(
				x=xaxis,
				y=templates[i],
				mode="lines"
			))

			new_center_line = {'x0': shift, 'x1': shift, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'y domain', 'line': {'dash': "dash", 'color': "Crimson"}}  #, 'annotation': {'text': "New Center", 'textangle': -90}}
			if i == 0:
				fig.add_shape(**new_center_line)

			labels.append(f"Unit {unit_id}")
			args.append({
				'title_text': f"Unit {unit_id}",
				'shapes': [old_center_line, new_center_line]
			})

		utils.plot_sliders(fig, 1, labels, f"{self.logs_folder}/alignment", args)
