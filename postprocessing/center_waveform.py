import os
from multiprocessing import Pool
import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phy_data import PhyData
import postprocessing.filter as filter
import postprocessing.utils as utils


def center_units(data: PhyData, unit_ids: list, params: dict, plot_folder: str):
	"""
	Uses a custom algorithm to recenter all spikes' waveform to the spike time.
	This should make the mean much nicer as the spikes will be more aligned.

	@param data:
		The data object.
	@param unit_ids (list):
		List of units' id to center the waveforms of.
	@param params (dict):
		Parameters for the center_waveform function.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	kwargs = {
		"unit_ids": unit_ids,
		"filter": [2, 150, 6000],
		"max_change": [1.5, 1.5],
		"ref_window": [1.5, 1.5],
		"ref_samples": 100,
		"ref_max_shift": 20,
		"waveforms": {
			"ms_before": None,
			"ms_after": None,
			"max_spikes_per_unit": None,
			"max_channels_per_waveforms": 5
		},
		"n_process": 8,
		"sampling_f": data.sampling_f,
		"uV_ratio": data.uvolt_ratio
	}
	kwargs.update(params)
	kwargs['waveforms']['ms_before'] = kwargs['max_change'][0] + kwargs['ref_window'][0]
	kwargs['waveforms']['ms_after']  = kwargs['max_change'][1] + kwargs['ref_window'][1]
	kwargs['waveforms']['max_spikes_per_unit'] = None
	kwargs['waveforms']['return_idx'] = True

	ms_to_pts = int(round(1e-3 * data.sampling_f))
	kwargs['max_change'] = (np.array(kwargs['max_change']) * ms_to_pts).round().astype(np.int16)
	kwargs['ref_window'] = (np.array(kwargs['ref_window']) * ms_to_pts).round().astype(np.int16)

	waveforms = data.get_units_waveforms(unit_ids, **kwargs['waveforms'])
	channels_idx = [w[1] for w in waveforms]
	waveforms = [w[0] for w in waveforms]
	old_means = np.array([np.mean(w[:, 0, kwargs['max_change'][0]:-kwargs['max_change'][1]], axis=0) for w in waveforms], dtype=np.float32)
	old_means2 = np.array([np.mean(w[:, :, kwargs['max_change'][0]:-kwargs['max_change'][1]], axis=0) for w in waveforms], dtype=np.float32)
	old_autocorrs = [utils.get_autocorr(data, unit_id, kwargs['plot']['auto_corr']['bin_size'], kwargs['plot']['auto_corr']['max_time'])[0] for unit_id in unit_ids]

	b, a = filter.get_filter_params(*kwargs['filter'], data.sampling_f)
	waveforms = filter.filter_units_waveforms_parallel(waveforms, b, a, dtype=np.int32, n_process=kwargs['n_process'])

	references = make_units_reference_parallel(waveforms, params=kwargs, n_process=kwargs['n_process'])
	shifts = center_units_waveforms_on_refs_parallel(waveforms, references, params=kwargs, plot_folder=plot_folder, unit_ids=unit_ids, n_process=kwargs['n_process'])

	for i in range(len(waveforms)):
		data.change_spikes_time(unit_ids[i], shifts[i])

	_plot_result(data, unit_ids, old_means, references[:, 0], old_autocorrs, kwargs, plot_folder)
	waveforms =  [data.get_unit_waveforms(unit_ids[i], **kwargs['waveforms'])[0] for i in range(len(unit_ids))]
	new_means = np.array([np.mean(w[:, :, kwargs['max_change'][0]:-kwargs['max_change'][1]], axis=0) for w in waveforms], dtype=np.float32)
	np.save("{0}/results/old_means.npy".format(plot_folder), old_means2)
	np.save("{0}/results/references.npy".format(plot_folder), references)
	np.save("{0}/results/new_means.npy".format(plot_folder), new_means)
	data.clear_wvfs()


def make_units_reference(units_waveforms: list, params: dict):
	"""
	Calls make_custom_reference for all units on a random subsample of waveforms.

	@param units_waveforms (list of np.ndarray) [n_units][n_waveforms, n_channels, time]:
		Units' waveforms used to make the reference (only a subsample will be considered).
	@param params (dict):
		Parameters for the center_waveform function.

	@return units_reference (np.ndarray[int32]) [n_units, n_channels, time]:
		Computed references for each unit.
	"""

	references = np.zeros([len(units_waveforms), units_waveforms[0].shape[1], np.sum(params['ref_window'])+1], np.int32)

	for i in range(len(units_waveforms)):
		unit_waveforms = units_waveforms[i]
		n = min(unit_waveforms.shape[0], params['ref_samples'])

		waveforms_idx = np.random.choice(np.arange(0, unit_waveforms.shape[0], 1), n, replace=False)
		waveforms = unit_waveforms[waveforms_idx]
		references[i] = make_custom_reference(waveforms, params)

	return references


def make_custom_reference(waveforms, params: dict):
	"""
	Creates a reference waveform from waveforms based on a custom method:
	First, we need to find the "best" waveform amongst all given waveforms. To achieve this,
	we iterate through each waveform, and for each iteration, we center all waveforms on this
	one. The sum of all shifts becomes the score for this waveform. At the end, the waveform
	with the lowest score (i.e. the least sum of shifts) is considered the best waveform.
	Second, we center each "good" waveform to this best waveform (a waveform is not good if
	its initial shift was greater than a certain value). The median of all these centered good
	waveforms make the temporary reference.
	We then center all of the "good" waveforms to this temporary reference, and take the median
	again, which is the custom reference.

	@param waveforms (np.ndarray) [n_waveforms, n_channels, time]
		Waveforms to use to make the reference.
	@param params (dict):
		Parameters for the center_waveform function.

	@return custom_reference (np.ndarray[int32]) [n_channels, time]:
		Computed reference from given waveforms.
	"""

	shifts = np.zeros([len(waveforms), len(waveforms)], dtype=np.int16)
	for i in range(len(waveforms)):
		shifts[i] = center_waveforms_on_ref(waveforms, waveforms[i, :, params['max_change'][0]:-params['max_change'][1]], params)

	sum_shifts = np.sum(np.abs(shifts), axis=1)
	best_wvf_idx = np.argmin(sum_shifts)

	idx_sorted = np.argsort(np.abs(shifts[best_wvf_idx]))
	a = np.argmax(idx_sorted == best_wvf_idx)		# With argsort, best_wvf_idx might not be in the first slot.
	idx_sorted[[0, a]] = idx_sorted[[a, 0]]

	if np.abs(shifts[best_wvf_idx, idx_sorted[-1]]) <= params['ref_max_shift']:
		n_good_waveforms = len(waveforms)
	else:
		n_good_waveforms = np.argmax(np.abs(shifts[best_wvf_idx, idx_sorted]) > params['ref_max_shift'])
	good_waveforms = np.zeros([n_good_waveforms, waveforms.shape[1], np.sum(params['ref_window'])+1], dtype=np.int32) # TODO: Best method?
	for i in range(n_good_waveforms):
		idx = idx_sorted[i]
		shift = shifts[best_wvf_idx, idx]
		good_waveforms[i] = waveforms[idx, :, params['max_change'][0]+shift:-params['max_change'][1]+shift]

	tmp_ref = np.median(good_waveforms, axis=0).round().astype(np.int32)
	centers = center_waveforms_on_ref(waveforms[idx_sorted[:n_good_waveforms]], tmp_ref, params)

	centered_waveforms = np.zeros([n_good_waveforms, waveforms.shape[1], np.sum(params['ref_window'])+1], dtype=np.int32)
	for i in range(n_good_waveforms):
		shift = centers[i]
		if shift > params['ref_max_shift']:
			continue

		centered_waveforms[i] = waveforms[idx_sorted[i], :, params['max_change'][0]+shift:-params['max_change'][1]+shift]

	return np.median(centered_waveforms, axis=0).round().astype(np.int32)


def center_units_waveforms_on_refs(units_waveforms: list, references, params: dict, plot_folder: str=None, unit_ids: list=None):
	"""
	Centers all units' waveforms on their respective reference.
	Returns the shifts computed from this centering.

	@param units_waveforms (list of np.ndarray) [n_units][n_waveforms, n_channels, time]:
		Units' waveforms to center on their respective reference.
	@param references (np.ndarray) [n_units, n_channels, time]:
		Each unit's reference waveform used for centering.
	@param params (dict):
		Parameters for the center_waveform function.
	@param plot_folder (str):
		Path to the plot folder. If None (default), will not plot.
	@param unit_id (list of int) [n_units]:
		Units ID for plotting. If None (default), will not plot.

	@return shift_centers (list of np.ndarray[int16]) [n_units][n_waveforms]:
		Computed shifts for all units' waveforms (negative means go back in time, 0 means do not change center).
	"""
	assert len(units_waveforms) == len(references)

	shifts = []
	for i in range(len(units_waveforms)):
		unit_id = unit_ids[i] if isinstance(unit_ids, (list, np.ndarray)) else None
		shifts.append(center_waveforms_on_ref(units_waveforms[i], references[i], params, plot_folder, unit_id))

	return shifts


def center_waveforms_on_ref(waveforms: np.ndarray, reference: np.ndarray, params: dict, plot_folder: str=None, unit_id: int=None):
	"""
	Returns new centers for the waveforms based on a reference.
	First, centers using convolusion with baseline substraction that aligns well on several ms but not on a sub ms scale.
	Validates this new centers with a threshold estimation for "good spikes".
	Then recenters to have the sub ms component using convolution without baseline substration.
	This is done using all channels.

	@param waveforms (np.ndarray) [n_waveforms, n_channels, time]:
		Waveforms to align. Warning: if type too small (e.g. int16), will probably result in a silent error.
	@param reference (np.ndarray) [n_channels, time]:
		Reference used for waveform alignment.
	@param params (dict):
		Parameters for the center_waveform function.
	@param plot_folder (str):
		Path to the plot folder. If None (default), will not plot.
	@param unit_id (int):
		Unit ID for plotting. If None (default), will not plot.

	@return shift_centers (np.ndarray[int16]) [n_waveforms]:
		Computed shifts for all waveforms (negative means go back in time, 0 means do not change center).
	"""

	conv = compute_convolution(waveforms, reference, substract_baseline=False)

	shifts = np.argmin(conv, axis=1) - params['max_change'][0]
	threshold = _compute_convolution_threshold(conv, reference, shifts)

	centers = _update_centers_threshold(-conv, -threshold, params['max_change'][0])
	if plot_folder != None and unit_id != None:
		_plot_center_on_ref(waveforms[:20, 0], reference[0], conv, threshold, unit_id, centers, params, plot_folder)

	return centers


def compute_convolution(waveforms: np.ndarray, reference: np.ndarray, substract_baseline: bool=False):
	"""
	Computes the convolution between a set of waveforms and a reference waveform.
	
	@param waveforms (np.ndarray) [n_waveforms, n_channels, time]:
		Waveforms to convolve. Warning: if type too small (e.g. int16), will probably result in an error.
	@param reference (np.ndarray) [n_channels, time]:
		Reference for convolution.
	@param substract_baseline (bool):
		If true, will substract the convolution with baseline over channels before summing.

	@return convolution (np.ndarray) [n_waveforms, time]:
		Result of convolution per waveform.
	"""

	conv = scipy.signal.fftconvolve(waveforms, reference[None, :, ::-1], mode="valid", axes=2)

	if substract_baseline:
		baseline = scipy.signal.fftconvolve(reference, reference[:, ::-1], mode="valid", axes=1)[:, 0]
		return np.sum((conv - baseline[None, :, None])**2, axis=1)
	else:
		return np.sum(conv, axis=1)


def _compute_convolution_threshold(conv: np.ndarray, reference: np.ndarray, shifts: np.ndarray, max_shift: int=15):
	"""
	Computes the threshold under which the convolution is considered "fiding a good spike".

	@param conv (np.ndarray) [n_waveforms, time]:
		Result of convolution (in order to compute an adapted threshold).
	@param shifts (np.ndarray) [n_waveforms]:
		Shifts to have new center if we stopped there.
	@param max_shift (int):
		Take only convolutions where shifts are under or equal to this value to compute threshold.
		A good treshold should only be calculated from good waveforms.

	@return threshold (float):
		Threshold under which a convolution is thought to represent a "good spike".
	"""

	conv = conv[np.where(np.abs(shifts) <= max_shift)]
	maximums = np.max(conv, axis=1)
	
	baseline = np.sum(scipy.signal.fftconvolve(reference, reference[:, ::-1], mode="valid", axes=1)[:, 0])
	maximums[maximums > baseline] = baseline

	std = scipy.stats.median_abs_deviation(maximums, scale="normal")

	return baseline - 3.5*std


def _update_centers_threshold(conv: np.ndarray, threshold: float, prev_center: int):
	"""
	Computes the new centers based on the nearest peak crossing the threshold.
	If never crosses the threshold, takes the lowest peak.

	@param conv (np.ndarray) [n_waveforms, time]:
		Result of convolution (to check against treshold).
	@param threshold (float):
		Threshold under which we consider to have a "good spike".
	@param prev_center (int):
		Center before any computation took place (the one in phy).

	@return shift_centers (np.ndarray[int16]) [n_waveforms]:
		New shifts from prev_center.
	"""

	centers = []

	for i in range(len(conv)):
		local_minima = scipy.signal.find_peaks(-conv[i], height=-threshold)[0]

		if local_minima.shape[0] == 0: # The convolution never crossed the threshold
			centers.append(np.argmin(conv[i]))
		else:
			centers.append(local_minima[np.argmin(np.abs(local_minima-prev_center))])

	return np.array(centers, dtype=np.int16) - prev_center




def make_units_reference_parallel(units_waveforms: list, params: dict, n_process: int=8):
	"""
	Parallel implementation of center_waveform.make_units_reference().
	"""
	if n_process == 1:
		return make_units_reference(units_waveforms, params)

	references = np.zeros([len(units_waveforms), units_waveforms[0].shape[1], np.sum(params['ref_window'])+1], np.int32)
	with Pool(processes=n_process) as pool:
		res = []

		for i in range(len(units_waveforms)):
			unit_waveforms = units_waveforms[i]
			n = min(unit_waveforms.shape[0], params['ref_samples'])

			waveforms_idx = np.random.choice(np.arange(0, unit_waveforms.shape[0], 1), n, replace=False)
			waveforms = unit_waveforms[waveforms_idx]

			res.append(pool.apply_async(make_custom_reference, (waveforms, params)))

		for i in range(len(units_waveforms)):
			references[i] = res[i].get()
	
	return references


def center_units_waveforms_on_refs_parallel(units_waveforms, references, params: dict, plot_folder: str=None, unit_ids: list=None, n_process: int=8):
	"""
	Parallel implementation of center_waveform.center_units_waveforms_on_refs().
	"""
	assert len(units_waveforms) == len(references)
	if n_process == 1:
		return center_units_waveforms_on_refs(units_waveforms, references, params)

	shifts = []
	with Pool(processes=n_process) as pool:
		res = []

		for i in range(len(units_waveforms)):
			unit_id = unit_ids[i] if isinstance(unit_ids, (list, np.ndarray)) else None
			res.append(pool.apply_async(center_waveforms_on_ref, (units_waveforms[i], references[i], params, plot_folder, unit_id)))

		for i in range(len(units_waveforms)):
			shifts.append(res[i].get())

	return shifts




def _plot_center_on_ref(waveforms: np.ndarray, reference: np.ndarray, conv, threshold, unit_id: int, centers: list, params: dict, plot_folder: str):
	"""

	"""

	save_folder = "{0}/centering".format(plot_folder)
	os.makedirs(save_folder, exist_ok=True)

	fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

	steps = []
	xaxis1 = np.arange(-params['ref_window'][0] - params['max_change'][0], params['ref_window'][1] + params['max_change'][1] + 1) * (1e3 / params['sampling_f'])
	xaxis2 = np.arange(-params['ref_window'][0], params['ref_window'][1]+1) * (1e3 / params['sampling_f'])
	xaxis3 = np.arange(-params['max_change'][0], params['max_change'][1]+1) * (1e3 / params['sampling_f'])

	for i in range(len(waveforms)):
		center = centers[i] * 1e3 / params['sampling_f']

		shapes = []
		shapes.append(dict(type="line",
			x0=0, x1=0, y0=0, y1=1, xref="x", yref="paper",
			line=dict(color="LightGreen", dash="dashdot")))
		shapes.append(dict(type="line",
			x0=center, x1=center, y0=0, y1=1, xref="x", yref="paper",
			line=dict(color="Crimson")))

		fig.add_trace(go.Scatter(
			x=xaxis1,
			y=waveforms[i] * params['uV_ratio'],
			mode="lines",
			marker_color="CornflowerBlue",
			name="Waveform",
			visible=False
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis2,
			y=reference * params['uV_ratio'],
			mode="lines",
			marker_color="OrangeRed",
			name="Reference",
			visible=False
		), row=1, col=1)
		"""fig.add_trace(go.Scatter(
			x=xaxis1,
			y=waveforms[i] * params['uV_ratio'],
			mode="lines",
			marker_color="CornflowerBlue",
			name="Waveform",
			visible=False
		), row=1, col=2)
		fig.add_trace(go.Scatter(
			x=xaxis2+center2,
			y=reference * params['uV_ratio'],
			mode="lines",
			marker_color="OrangeRed",
			name="Reference",
			visible=False
		), row=1, col=2)"""
		fig.add_trace(go.Scatter(
			x=xaxis3,
			y=conv[i],
			mode="lines",
			marker_color="CornflowerBlue",
			name="First convolution",
			visible=False
		), row=2, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis3,
			y=[threshold]*len(xaxis3),
			mode="lines",
			marker_color="DarkViolet",
			name="Threshold",
			visible=False
		), row=2, col=1)
		"""fig.add_trace(go.Scatter(
			x=np.arange(-len(conv2[i])//2+center2-center1, len(conv2[i]//2+1+center2-center1)) * (1e3 / params['sampling_f']),
			y=conv2[i],
			mode="lines",
			marker_color="CornflowerBlue",
			name="Second convolution",
			visible=False
		), row=2, col=2)"""

		step = dict(
			label="{0}".format(i+1),
			method="update",
			args=[
				{"visible": [j//4 == i for j in range(4*len(waveforms))]},
				{"title.text": "Waveform {0}".format(i+1),
				"shapes": shapes}
			]
		)
		steps.append(step)

	for i in range(4):
		fig.data[i].visible = True
	sliders = [dict(
		active=0,
		currentvalue={"prefix": "Waveform "},
		pad={"t": 50},
		steps=steps
	)]

	fig.update_yaxes(title_text="Voltage (µV)", row=1, col=1)
	# fig.update_yaxes(title_text="Voltage (µV)", row=1, col=2)
	fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
	# fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
	fig.update_yaxes(type="log", row=2, col=1)
	fig.update_layout(width=1440, height=810, sliders=sliders)
	fig.write_html("{0}/unit_{1}.html".format(save_folder, unit_id))


def _plot_result(data: PhyData, unit_ids: list, old_means: np.ndarray, references: np.ndarray, old_autocorrs: list, params: dict, plot_folder: str):
	"""
	Plots the result of centering (old mean, new mean, reference waveforms, as well as pre/post auto-correlograms).

	@param data (PhyData):
		The data object.
	@param unit_ids (list):
		List of units' id to plot.
	@param old_means (np.ndarray) [n_units, time]:
		The mean waveform of each unit pre-centering.
	@param references (np.ndarray) [n_units, time]:
		The reference waveform used of each unit.
	@param old_autocorrs (list of np.ndarray) [n_units][time]:
		List of each unit's auto-correlogram pre-centering.
	@param params (dict):
		Parameters for the center_waveform function.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	m = params['plot']['auto_corr']['max_time'] - params['plot']['auto_corr']['bin_size'] / 2
	save_folder = "{0}/results".format(plot_folder)
	os.makedirs(save_folder, exist_ok=True)

	new_autocorrs = [utils.get_autocorr(data, unit_id, params['plot']['auto_corr']['bin_size'], params['plot']['auto_corr']['max_time'])[0] for unit_id in unit_ids]
	waveforms = data.get_units_waveforms(unit_ids, ms_before=params['waveforms']['ms_before'], ms_after=params['waveforms']['ms_after'], max_channels_per_waveforms=1)
	new_means = np.array([np.mean(w[:, 0, params['max_change'][0]:-params['max_change'][1]], axis=0) for w in waveforms], dtype=np.float32)
	del waveforms

	fig, axes = plt.subplots(2)
	xaxis = np.linspace(-params['waveforms']['ms_before'], params['waveforms']['ms_after'], references.shape[1])

	old_mean_plot, = axes[0].plot(xaxis, old_means[0], label="Old mean")
	reference_plot, = axes[0].plot(xaxis, references[0], label="Reference")
	new_mean_plot, = axes[0].plot(xaxis, new_means[0], label="New mean")
	axes[0].legend()
	axes[0].set_xlabel("Time (ms)", fontsize=14)
	axes[0].set_ylabel("Voltage (A.U.)", fontsize=14)

	xaxis = np.linspace(-m, m, new_autocorrs[0].shape[0])
	old_autocorr_plot, = axes[1].plot(xaxis, old_autocorrs[0], label="Old auto-corr")
	new_autocorr_plot, = axes[1].plot(xaxis, new_autocorrs[0], label="New auto-corr")
	axes[1].legend()
	axes[1].set_xlabel("Time (ms)", fontsize=14)

	for i in range(len(unit_ids)):
		unit_id = unit_ids[i]
		fig.suptitle("Sorting {0}, unit {1}".format(data.sorting_idx, unit_id), fontsize=24)

		old_mean_plot.set_ydata(old_means[i])
		reference_plot.set_ydata(references[i])
		new_mean_plot.set_ydata(new_means[i])
		axes[0].relim()
		axes[0].autoscale_view()

		old_autocorr_plot.set_ydata(old_autocorrs[i])
		new_autocorr_plot.set_ydata(new_autocorrs[i])
		axes[1].relim()
		axes[1].autoscale_view()

		fig.canvas.draw()
		fig.canvas.flush_events()

		fig.savefig("{0}/unit_{1}.png".format(save_folder, unit_id))
	
	plt.close("all")

