import os
import numpy as np
import scipy.signal
import plotly.graph_objects as go

from phy_data import PhyData
import postprocessing.filter as filter


def center_units(data: PhyData, unit_ids: list, params: dict, plot_folder: str):
	"""
	Uses a custom algorithm to recenter the mean waveform so that the peak corresponds to the spike center.
	This helps when comparing spikes between units as there won't be a shift.

	@param data:
		The data object.
	@param unit_ids (list):
		List of units' id to center the mean waveform of.
	@param params (dict):
		Parameters for the center_cluster function.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	kwargs = {
		"unit_ids": unit_ids,
		"waveforms": {
			"ms_before": 1.5,
			"ms_after": 1.5,
			"max_spikes_per_unit": 10000,
			"max_channels_per_waveforms": 5
		},
		"filter": [2, 300, 6000],
		"threshold_std": 1.5,
		"check_next": 10
	}
	kwargs.update(params)

	waveforms = data.get_units_waveforms(unit_ids, **kwargs['waveforms'])
	waveforms = np.array([np.mean(w, axis=0, dtype=np.float32) for w in waveforms])

	b, a = filter.get_filter_params(*kwargs['filter'], data.sampling_f)
	filtered_wvfs = filter.filter(waveforms, b, a, dtype=np.float32)
	channels = np.argmax(np.amax(np.abs(filtered_wvfs), axis=2), axis=1)

	centers = np.zeros([len(waveforms)], dtype=np.int16)
	wvfs = []
	for i in range(len(waveforms)):
		centers[i] = get_shift_unit(filtered_wvfs[i, channels[i]], kwargs, data.sampling_f)
		data.change_spikes_time(unit_ids[i], centers[i])
		wvfs.append(filtered_wvfs[i, channels[i]])

	wvfs = np.array(wvfs, dtype=np.float32)
	_plot_result(data, wvfs, unit_ids, centers, kwargs, plot_folder)
	data.clear_wvfs()


def get_shift_unit(mean_waveform: np.ndarray, params: dict, sampling_f: float, plot=False):
	"""
	Returns the shift necessary to all of the unit's waveforms in order to center the unit's spike times.
	Uses only the best channel, takes the mean and filters it before calling center_cluster.compute_center().

	@param waveforms (np.ndarray) [time]:
		Unit's mean waveform on best channel.
	@param params (dict):
		parameters of the center_cluster function.
	@param sampling_f (float):
		Sampling rate of the recording.

	@return unit_shift (int):
		Shift needed on all of the unit's waveforms to center the unit's mean waveform.
	"""

	threshold = params['threshold_std'] * np.std(mean_waveform)
	center = compute_center(mean_waveform, threshold, check_next=params['check_next'])

	return int(round(center - params['waveforms']['ms_before'] * sampling_f * 1e-3))


def compute_center(waveform: np.ndarray, threshold: float, check_next: int=10):
	"""
	Finds the center of a waveform by finding the first peak higher than a threshold.
	After finding it, checks the next 'check_next' timepoints for a higher peak.
	This avoids taking a local extrema, while still taking the first 'big' peak.

	@param waveform (np.ndarray) [time]:
		Waveform to find the center of.
	@param threshold (float):
		Threshold to cross to consider a peak a real one (will also consider -threshold).
	@param check_next (int):
		See description above.

	@return center (int):
		Center of the waveform as an index of the waveform array.
	"""

	maxima = scipy.signal.find_peaks(waveform, height=threshold)
	minima = scipy.signal.find_peaks(-waveform, height=threshold)

	if maxima[0].shape[0] > 0:
		idx = np.sum(maxima[0] < maxima[0][0]+check_next)
		maximum = maxima[0][np.argmax(maxima[1]['peak_heights'][:idx])]
	else:
		maximum = 1e7

	if minima[0].shape[0] > 0:
		idx = np.sum(minima[0] < minima[0][0]+check_next)
		minimum = minima[0][np.argmax(minima[1]['peak_heights'][:idx])]
	else:
		minimum = 1e7

	center = min(maximum, minimum)
	if center < 9e6:
		return center

	# If no peak was found, redo with threshold/2
	return compute_center(waveform, threshold/2)




def _plot_result(data: PhyData, mean_waveforms: np.ndarray, unit_ids: list, centers: list, params: dict, plot_folder: str):
	"""
	Plots the result of centering (old center vs new center).

	@param data (PhyData):
		The data object.
	@param mean_waveforms (np.ndarray) [n_units, time]:
		The mean waveform of each unit pre-centering.
	@param unit_ids (list):
		List of units' id to plot.
	@param centers (list of int) [n_units]:
		New center relative to old center of each unit (in sampling time).
	@param params (dict):
		Parameters for the center_waveform function.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	os.makedirs(plot_folder, exist_ok=True)

	fig = go.Figure()
	steps = []
	xaxis = np.linspace(-params['waveforms']['ms_before'], params['waveforms']['ms_after'], mean_waveforms.shape[1])
	
	fig.add_vline(x=0, line_color="goldenrod", line_dash="dash", annotation_text="Old center", annotation_textangle=-90)
	fig.update_xaxes(title_text="Time (ms)")
	fig.update_yaxes(title_text="Voltage (A.U.)")

	for i in range(len(unit_ids)):
		unit_id = unit_ids[i]

		fig.add_trace(go.Scatter(x=xaxis, y=mean_waveforms[i], mode="lines", name="Mean waveform".format(unit_id), visible=False))
		fig.add_trace(go.Scatter(x=[centers[i]*1e3/data.sampling_f]*2, y=[np.min(mean_waveforms[i]), np.max(mean_waveforms[i])], mode="lines", marker_color="red", name="New center", visible=False))

		step = dict(
			label=str(unit_id),
			method="update",
			args=[
				{"visible": [j//2 == i for j in range(2*len(unit_ids))]},
				{"title": "Sorting {0}, unit {1}".format(data.sorting_idx, unit_id)}
			]
		)
		steps.append(step)

	fig.data[0].visible = True
	fig.data[1].visible = True
	sliders = [dict(
		active=0,
		currentvalue={"prefix": "Unit "},
		pad={"t": 50},
		steps=steps
	)]

	fig.update_layout(width=1440, height=810, sliders=sliders)
	fig.write_html("{0}/results.html".format(plot_folder))
