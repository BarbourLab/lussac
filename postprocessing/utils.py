import os
from pathlib import Path
import math
import itertools
import ctypes
import numpy as np
import scipy.interpolate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import postprocessing.filter as filter


folder = Path(__file__).resolve().parent
cpp_file = "{0}/utils_cpp.so".format(folder)
assert os.path.exists(cpp_file), "The cpp code has not been compiled!\nRun the compile file in the main folder."
cpp = ctypes.CDLL(cpp_file)
c_uint64_p = ctypes.POINTER(ctypes.c_uint64)

cpp.compute_ISI.argtypes			= c_uint64_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
cpp.compute_autocorr.argtypes		= c_uint64_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
cpp.compute_crosscorr.argtypes		= c_uint64_p, c_uint64_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32
cpp.compute_firing_rate.argtypes	= c_uint64_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
cpp.compute_spikes_refractory_period.argtypes	= c_uint64_p, ctypes.c_uint32, ctypes.c_uint32
cpp.compute_nb_coincident_spikes.argtypes	= c_uint64_p, c_uint64_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32
cpp.compute_cross_refractory_period.argtypes= c_uint64_p, c_uint64_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32


def get_ISI(data, unit_id: int, bin_size: float=1.0, max_time: float=50.0):
	"""
	Computes the Inter-Spike Interval (time difference between consecutive spikes) of a unit.
	Takes ~4ms for 200k spikes.
	
	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param max_time (float):
		Stop the ISI histogram at this value (in ms).

	@return ISI (np.ndarray[uint32]) [time]:
		ISI histogram of the unit.
	@return bins (np.ndarray) [time+1]:
		Bins of the histogram.
	"""

	spike_train = data.get_unit_spike_train(unit_id=unit_id)
	return get_ISI_from_spiketrain(spike_train, bin_size, max_time, data.sampling_f)


def get_ISI_from_spiketrain(spike_train: np.ndarray, bin_size: float=1.0, max_time: float=50.0, sampling_f: float=3e4):
	"""
	Computes the Inter-Spike Interval (time difference between consecutive spikes) of a spike train.
	Takes ~4ms for 200k spikes.

	@param spike_train (np.ndarray[uint64]) [n_spikes]:
		Spike train to use (timings in sampling time).
	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param max_time (float):
		Stop the ISI histogram at this value (in ms).
	@param sampling_f (float):
		Sampling frequency of the recording (in Hz).

	@return ISi (np.ndarray[uint32]) [time]:
		ISI histogram of the spike train.
	@return bins (np.ndarray) [time+1]
		Bins of the histogram.
	"""

	if spike_train.dtype != np.uint64:
		spike_train = spike_train.astype(np.uint64)

	bin_size = int(round(bin_size * 1e-3 * sampling_f))
	max_time = int(round(max_time * 1e-3 * sampling_f))
	max_time -= max_time % bin_size
	bins = np.arange(0, max_time+bin_size, bin_size) * 1e3 / sampling_f

	cpp.compute_ISI.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_uint32, shape=(len(bins)-1,))
	ISI = cpp.compute_ISI(spike_train.ctypes.data_as(c_uint64_p), ctypes.c_uint32(len(spike_train)), ctypes.c_uint32(bin_size), ctypes.c_uint32(max_time))

	return (ISI, bins)


def get_autocorr(data, unit_id: int, bin_size: float=1.0/3.0, max_time: float=50.0):
	"""
	Computes the auto-correlogram (time difference between spikes, consecutive and non-consecutive) of a unit.
	Takes ~10ms for 200k spikes.

	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param max_time (float):
		Stop the autocorr histogram at this value (in ms).

	@return auto_correlogram (np.ndarray[uint32]) [time]:
		Auto-correlogram as a histogram.
	@return bins (np.ndarray) [time+1]:
		Bins of the histogram.
	"""

	spike_train = data.get_unit_spike_train(unit_id=unit_id)
	return get_autocorr_from_spiketrain(spike_train, bin_size, max_time, data.sampling_f)


def get_autocorr_from_spiketrain(spike_train: np.ndarray, bin_size: float=1.0/3.0, max_time: float=50.0, sampling_f: float=3e4):
	"""
	Computes the auto-correlogram (time difference between spikes, consecutive and non-consecutive) of a spike train.
	Takes ~10ms for 200k spikes.

	@param spike_train (np.ndarray[uint64]) [n_spikes]:
		Spike train to use (timings in sampling time).
	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param max_time (float):
		Stop the autocorr histogram at this value (in ms).
	@param sampling_f (float):
		Sampling frequency of the recording (in Hz).

	@return ISi (np.ndarray[uint32]) [time]:
		Auto-correlogram as a histogram.
	@return bins (np.ndarray) [time+1]
		Bins of the histogram.
	"""

	if spike_train.dtype != np.uint64:
		spike_train = spike_train.astype(np.uint64)

	bin_size = int(round(bin_size * 1e-3 * sampling_f))
	max_time = int(round(max_time * 1e-3 * sampling_f))
	max_time -= max_time % bin_size
	bins = np.arange(-max_time, max_time+bin_size, bin_size) * 1e3 / sampling_f

	cpp.compute_autocorr.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_uint32, shape=(len(bins)-1,))
	auto_corr = cpp.compute_autocorr(spike_train.ctypes.data_as(c_uint64_p), ctypes.c_uint32(len(spike_train)), ctypes.c_uint32(bin_size), ctypes.c_uint32(max_time))

	return (auto_corr, bins)


def get_crosscorr(data, unit1_id: int, unit2_id: int, bin_size: float=1.0/3.0, max_time: float=30.0):
	"""
	Computes the cross-correlogram (time difference between two spike trains) between two units.
	Takes ~3ms for 200k - 3.5k spikes.

	@param unit1_id (int), unit2_id (int):
		ID of both units containing the spike trains.
	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param max_time (float):
		Stop the crosscorr histogram at this value (in ms).

	@return cross_correlogram (np.ndarray[uint32]) [time]:
		Cross-correlogram as a histogram.
	@return bins (np.ndarray) [time+1]:
		Bins of the histogram.
	"""

	
	spike_train1 = data.get_unit_spike_train(unit_id=unit1_id)
	spike_train2 = data.get_unit_spike_train(unit_id=unit2_id)

	return get_crosscorr_from_spiketrain(spike_train1, spike_train2, bin_size, max_time, data.sampling_f)


def get_crosscorr_from_spiketrain(spike_train1: np.ndarray, spike_train2: np.ndarray, bin_size: float=1.0/3.0, max_time: float=30.0, sampling_f: float=3e4):
	"""
	Computes the cross-correlogram (time difference between two spike trains).
	Takes ~3ms for 200k - 3.5k spikes.

	@param spike_train1 (np.ndarray[uint64]) [n_spikes1]:
		First spike train to use (timings in sampling time).
	@param spike_train2 (np.ndarray[uint64]) [n_spikes2]:
		Second spike train to use (timings in sampling time).
	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param max_time (float):
		Stop the crosscorr histogram at this value (in ms).
	@param sampling_f (float):
		Sampling frequency of the recording (in Hz).

	@return ISi (np.ndarray[uint32]) [time]:
		Cross-correlogram as a histogram.
	@return bins (np.ndarray) [time+1]
		Bins of the histogram.
	"""

	if spike_train1.dtype != np.uint64:
		spike_train1 = spike_train1.astype(np.uint64)
	if spike_train2.dtype != np.uint64:
		spike_train2 = spike_train2.astype(np.uint64)

	bin_size = int(round(bin_size * 1e-3 * sampling_f))
	max_time = int(round(max_time * 1e-3 * sampling_f))
	max_time -= max_time % bin_size
	bins = np.arange(-max_time, max_time+bin_size, bin_size) * 1e3 / sampling_f

	cpp.compute_crosscorr.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_uint32, shape=(len(bins)-1,))
	cross_corr = cpp.compute_crosscorr(spike_train1.ctypes.data_as(c_uint64_p), spike_train2.ctypes.data_as(c_uint64_p), ctypes.c_uint32(len(spike_train1)), ctypes.c_uint32(len(spike_train2)), ctypes.c_int32(bin_size), ctypes.c_int32(max_time))

	return (cross_corr, bins)


def get_firing_rate(data, unit_id: int, bin_size: float=2.0, as_Hz: bool=False):
	"""
	Computes the firing rate over time of a unit.

	@param as_Hz (bool):
		False = returns number of spikes in bin.
		True = returns frequency in bin.
	@param bin_size (float):
		Size of bin for histogram (in s).

	@return firing_rate (np.ndarray[uint32]) [time]:
		Firing rate as a histogram.
	@return bins (np.ndarray) [time+1]:
		Bins of the histogram.
	"""

	spike_train = data.get_unit_spike_train(unit_id=unit_id)
	return get_firing_rate_from_spiketrain(spike_train, data.recording.get_num_frames(), bin_size, as_Hz, data.sampling_f)


def get_firing_rate_from_spiketrain(spike_train: np.ndarray, t_max: int, bin_size: float=2.0, as_Hz: bool=False, sampling_f: float=3e4):
	"""
	Computes the firing rate over time of a spike train.

	@param spike_train (np.ndarray[uint64]) [n_spikes]:
		Spike train to use (timings in sampling time).
	@param t_max (int):
		Number of "frames" in the recording (in sampling time).
	@param bin_size (float):
		Size of bin for the histogram (in ms).
	@param as_Hz (bool):
		False = returns number of spikes in bin.
		True = returns frequency in bin.
	@param sampling_f (float):
		Sampling frequency of the recording (in Hz).

	@return firing_rate (np.ndarray[uint32]) [time]:
		Firing rate as a histogram.
	@return bins (np.ndarray) [time+1]:
		Bins of the histogram.
	"""

	if spike_train.dtype != np.uint64:
		spike_train = spike_train.astype(np.uint64)

	bin_size_c = int(round(bin_size * sampling_f))
	bins = (np.arange(0, t_max+bin_size_c, bin_size_c, dtype=np.uint64) / sampling_f).astype(np.float64)

	cpp.compute_firing_rate.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_uint32, shape=(len(bins)-1,))
	firing_rate = cpp.compute_firing_rate(spike_train.ctypes.data_as(c_uint64_p), ctypes.c_uint32(len(spike_train)), ctypes.c_uint32(bin_size_c), ctypes.c_uint32(len(bins)-1))

	if as_Hz:
		firing_rate = firing_rate.astype(np.float64) / bin_size

	return (firing_rate, bins)


def estimate_unit_contamination(data, unit_id: int, refractory_period: tuple=(0.0, 1.0)):
	"""
	Estimates the contamination of a unit based on the refractory period.
	Takes ~1.5ms for 200k spikes.

	@param refractory_period (tuple 2 float):
		Lower and upper bound of the refractory period (in ms, excluded).

	@return contamination (float between 0 and 1):
		Estimated contamination ratio.
	"""

	spike_train = data.get_unit_spike_train(unit_id=unit_id)
	t_max = data.recording.get_num_frames()

	return estimate_spike_train_contamination(spike_train, refractory_period, t_max, data.sampling_f)


def estimate_spike_train_contamination(spike_train: np.ndarray, refractory_period: tuple, t_max: int, sampling_f: float):
	"""
	Estimate the contamination of a spike train based on the refractory period.
	Takes ~1.5ms for 200k spikes.

	@param spike_train (np.ndarray[uint64]) [n_spikes]:
		Timings of each spikes (in sampling time).
	@param refractory_period (tuple of 2 floats):
		Window of the refractory period (in ms).
	@param t_max (int):
		Number of sampling time in recording.
	@param sampling_f (float):
		Sampling frequency of recording (in Hz).

	@return contamination (float between 0 and 1):
		Estimated contamination ratio.
	"""

	if spike_train.dtype != np.uint64:
		spike_train = spike_train.astype(np.uint64)

	lower_b = int(round(refractory_period[0] * 1e-3 * sampling_f))
	upper_b = int(round(refractory_period[1] * 1e-3 * sampling_f))
	t_r = upper_b - lower_b
	t_c = lower_b
	N = len(spike_train)
	t_max = t_max - 2*N*t_c

	n_v = cpp.compute_spikes_refractory_period(spike_train.ctypes.data_as(c_uint64_p), ctypes.c_uint32(len(spike_train)), ctypes.c_uint32(upper_b))

	A = n_v * t_max / (N**2 * t_r)
	if A > 1:
		return 1.0

	return 1.0 - math.sqrt(1.0 - A)


def estimate_units_contamination(data, unit_ids: list, refractory_period: tuple=(0.0, 1.0)):
	"""
	Analog to estimate_contamination, but merges the spike train of multiple units
	to evaluate what the contamination ratio would be if those units were to me merged.
	"""

	spike_trains = [data.get_unit_spike_train(unit_id=unit_id) for unit_id in unit_ids]
	spike_train = np.sort(list(itertools.chain(*spike_trains))).astype(np.uint64)
	t_max = data.recording.get_num_frames()

	return estimate_spike_train_contamination(spike_train, refractory_period, t_max, data.sampling_f)


def get_nb_coincident_spikes(spike_train1: np.ndarray, spike_train2: np.ndarray, window: int=5):
	"""
	Returns the number of coincident spikes between two spike trains.

	@param spike_train1 (np.ndarray[uint64]) [n_spikes1]:
		First spike train to use (timings in sampling time).
	@param spike_train2 (np.ndarray[uint64]) [n_spikes2]:
		Second spike train to use (timings in sampling time).
	@param window (int):
		Window to consider spikes to be coincident (in sampling time). 0 = need to be exact same timing.

	@return int:
		Number of coincident spikes between the two spike trains.
	"""

	if spike_train1.dtype != np.uint64:
		spike_train1 = spike_train1.astype(np.uint64)
	if spike_train2.dtype != np.uint64:
		spike_train2 = spike_train2.astype(np.uint64)

	return cpp.compute_nb_coincident_spikes(spike_train1.ctypes.data_as(c_uint64_p), spike_train2.ctypes.data_as(c_uint64_p), ctypes.c_uint32(len(spike_train1)), ctypes.c_uint32(len(spike_train2)), ctypes.c_int32(window))


def estimate_cross_spiketrains_contamination(spike_train1: np.ndarray, spike_train2: np.ndarray, refractory_period: tuple, t_max: int, sampling_f: float):
	"""
	Estimates the contamination of a spike train based on the refractory period collision with another spike train.

	@param spike_train1 (np.ndarray[uint64]) [n_spikes1]:
		Timings of each spikes (in sampling time) for the spike train we want the contamination of.
	@param spike_train2 (np.ndarray[uint64]) [n_spikes2]:
		Timings of each spikes (in sampling time) for the "ground truth" spike train.
	@param t_max (int):
		Number of sampling time in recording.
	@param sampling_f (float):
		Sampling frequency of recording (in Hz).

	@return contamination (float)
		Estimated contamination ratio.
	"""

	if spike_train1.dtype != np.uint64:
		spike_train1 = spike_train1.astype(np.uint64)
	if spike_train2.dtype != np.uint64:
		spike_train2 = spike_train2.astype(np.uint64)

	lower_b = int(round(refractory_period[0] * 1e-3 * sampling_f))
	upper_b = int(round(refractory_period[1] * 1e-3 * sampling_f))
	t_r = upper_b - lower_b

	C = estimate_spike_train_contamination(spike_train2, refractory_period, t_max, sampling_f)
	nb_refract = cpp.compute_cross_refractory_period(spike_train1.ctypes.data_as(c_uint64_p), spike_train2.ctypes.data_as(c_uint64_p), ctypes.c_uint32(len(spike_train1)), ctypes.c_uint32(len(spike_train2)), ctypes.c_uint32(lower_b), ctypes.c_uint32(upper_b))

	estimate1 = len(spike_train1) * len(spike_train2)*C * 2*t_r / t_max
	estimate2 = len(spike_train1) * len(spike_train2)*(1-C) * 2*t_r / t_max

	return (nb_refract - estimate1) / estimate2


def get_unit_supression_period(data, unit_id: int, contamination: float, **kwargs):
	"""

	"""

	spike_train = data.get_unit_spike_train(unit_id=unit_id)
	t_max = data.recording.get_num_frames()

	return get_spiketrain_supression_period(spike_train, contamination, t_max, data.sampling_f, **kwargs)


def get_spiketrain_supression_period(spike_train: np.ndarray, contamination: float, t_max: int, sampling_f: float, bin_size: float=1.0/3.0, max_time: float=80.0, threshold: float=0.12, max_f: float=500):
	"""

	"""

	correlogram, bins = get_autocorr_from_spiketrain(spike_train, bin_size, max_time, sampling_f)
	baseline = 2 * len(spike_train)**2 * bin_size*1e-3 * sampling_f / t_max
	baseline_cont = 2 * len(spike_train)**2 * contamination * bin_size*1e-3 * sampling_f / t_max
	half_point = len(correlogram)//2

	b, a = filter.get_filter_params(2, max_f=max_f, sampling_rate=1e3/bin_size, btype="lowpass")
	correlogram = filter.filter(correlogram.astype(np.int32), b, a, np.float32) - baseline_cont
	correlogram = correlogram[half_point:]
	threshold = threshold * baseline

	suppression_period = bin_size * np.argmax(np.array([np.all(correlogram[i:] > threshold) for i in range(len(correlogram))], dtype=bool))
	if suppression_period < bin_size or suppression_period > max_time-bin_size:
		return suppression_period

	xaxis = (bins[half_point+1:] + bins[half_point:-1]) / 2
	f = scipy.interpolate.interp1d(xaxis, correlogram)
	for t in np.linspace(suppression_period-bin_size, suppression_period+bin_size, 100):
		if f(t) > threshold:
			return suppression_period

	return suppression_period




def plot_unit(data, unit_id: int, save_folder: str, bin_size: float=1.0/3.0, max_time: float=50.0, fr_bin_size: float=2.0, ms_before: float=2, ms_after: float=2):
	"""
	Plots the characteristics of a unit in a given folder.

	@param data (PhyData):
		The data object.
	@param unit_id (int):
		Unit's ID.
	@param save_folder (str):
		Path to the folder where the plot will be saved.
	@param bin_size (float):
		Size of bin for the ISI and auto-correlogram (in ms).
	@param max_time (float):
		Time limit of the ISI and auto-correlogram (in ms).
	@param fr_bin_size (float):
		Size of bin for the firing rate histogram (in s).
	@param ms_before, ms_after (float):
		Timings for the waveform extraction (in ms).
	"""

	ISI = get_ISI(data, unit_id, bin_size, max_time)
	auto_corr = get_autocorr(data, unit_id, bin_size, max_time)
	firing_rate = get_firing_rate(data, unit_id, fr_bin_size, as_Hz=True)

	waveforms = data.get_unit_waveforms(unit_id, ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=5000, max_channels_per_waveforms=3, return_idx=True)
	channels = waveforms[1]
	waveforms = np.mean(waveforms[0], axis=0)
	wvf_xaxis = np.linspace(-ms_before, ms_after, waveforms.shape[1])
	waveforms = (wvf_xaxis, waveforms, channels, data.uvolt_ratio)

	plot_unit_correlograms(unit_id, save_folder, ISI, auto_corr, firing_rate, waveforms)


def plot_unit_from_spiketrain(data, spike_train: np.ndarray, unit_id: int, save_folder: str, bin_size: float=1.0/3.0, max_time: float=50.0,
							  fr_bin_size: float=2.0, ms_before: float=2, ms_after: float=2):
	"""
	Plots the characteristics of a unit in a given folder.

	@param data (PhyData):
		The data object.
	@param spike_train (np.ndarray[uint64]) [n_spikes]:
		Spike train to use.
	@param unit_id (int):
		Unit's ID.
	@param save_folder (str):
		Path to the folder where the plot will be saved.
	@param bin_size (float):
		Size of bin for the ISI and auto-correlogram (in ms).
	@param max_time (float):
		Time limit of the ISI and auto-correlogram (in ms).
	@param fr_bin_size (float):
		Size of bin for the firing rate histogram (in s).
	@param ms_before, ms_after (float):
		Timings for the waveform extraction (in ms).
	"""

	ISI = get_ISI_from_spiketrain(spike_train, bin_size, max_time, data.sampling_f)
	auto_corr = get_autocorr_from_spiketrain(spike_train, bin_size, max_time, data.sampling_f)
	firing_rate = get_firing_rate_from_spiketrain(spike_train, data.get_num_frames(), fr_bin_size, True, data.sampling_f)

	waveforms = data.get_waveforms_from_spiketrain(spike_train, ms_before=ms_before, ms_after=ms_after, max_channels_per_waveforms=3, return_idx=True)
	channels = waveforms[1]
	waveforms = np.mean(waveforms[0], axis=0)
	wvf_xaxis = np.linspace(-ms_before, ms_after, waveforms.shape[1])
	waveforms = (wvf_xaxis, waveforms, channels, data.uvolt_ratio)

	plot_unit_correlograms(unit_id, save_folder, ISI, auto_corr, firing_rate, waveforms)


def plot_unit_correlograms(unit_id: int, save_folder: str, ISI: tuple, auto_corr: tuple, firing_rate: tuple, waveforms: tuple):
	"""
	Plots the characteristics of a unit in a given folder.

	@param unit_id (int):
		Unit's ID.
	@param save_folder (str):
		Path to the folder where the plot will be saved.
	@param ISI (tuple):
		Bins and correlogram for the ISI.
	@param auto_corr (tuple):
		Bins and correlogram of the auto-correlogram.
	@param firing_rate (tuple):
		Bins and correlogram of the firing rate (in Hz).
	@param waveforms (tuple):
		x-axis: np.ndarray (in ms).
		waveforms: np.ndarray [n_channels>=3, time].
		channel_idx: list [n_channels >= 3]
	"""

	os.makedirs(save_folder, exist_ok=True)
	
	fig = make_subplots(rows=3, cols=2, subplot_titles=("ISI", "Mean waveform", "Auto-correlogram", "", "Firing rate", ""))

	w = ISI[1][1] - ISI[1][0]
	fig.add_trace(go.Bar(
		x=ISI[1][:-1] + w/2,
		y=ISI[0],
		width=w,
		name="ISI"
	), row=1, col=1)
	fig.update_xaxes(title_text="Time (ms)", row=1, col=1)

	w = auto_corr[1][1] - auto_corr[1][0]
	fig.add_trace(go.Bar(
		x=auto_corr[1][:-1] + w/2,
		y=auto_corr[0],
		width=w,
		name="Auto-correlogram"
	), row=2, col=1)
	fig.update_xaxes(title_text="Time (ms)", row=2, col=1)

	w = firing_rate[1][1] - firing_rate[1][0]
	fig.add_trace(go.Bar(
		x=firing_rate[1][:-1] + w/2,
		y=firing_rate[0],
		width=w,
		name="Firing rate"
	), row=3, col=1)
	fig.update_xaxes(title_text="Time (s)", row=3, col=1)
	fig.update_yaxes(title_text="Firing rate (Hz)", row=3, col=1)

	for i in range(3):
		fig.add_trace(go.Scatter(
			x=waveforms[0],
			y=waveforms[1][i] * (1 if waveforms[3] is None else waveforms[3]),
			mode="lines",
			name="Channel {0}".format(waveforms[2][i])
		), row=i+1, col=2)
		fig.update_xaxes(title_text="Time (ms)", row=i+1, col=2)
		fig.update_yaxes(title_text="Voltage ({0})".format("A.U." if waveforms[3] is None else "µV"), row=i+1, col=2)

	fig.add_annotation(x=0.3, y=1.1, xref="paper", yref="paper", showarrow=False, text="Frequency = {0:.2f} Hz".format(np.mean(firing_rate[0])))

	fig.update_layout(width=1200, height=900, title_text="Unit {0}".format(unit_id))
	fig.write_html("{0}/unit_{1}.html".format(save_folder, unit_id))
	fig.write_image("{0}/unit_{1}.png".format(save_folder, unit_id))


def plot_units(data, unit_ids: list, save_folder: str, bin_size: float=1.0/3.0, max_time: float=50.0, fr_bin_size: float=2.0, ms_before: float=2, ms_after: float=2, filename: str="units"):
	"""
	Plots the characteristics of multiple units in a given folder (as one file).

	@param data (PhyData):
		The data object.
	@param unit_ids (list of int):
		List of units' ID.
	@param save_folder (str):
		Path to the folder where the plot will be saved.
	@param bin_size (float):
		Size of bin for the ISI and auto-correlogram (in ms).
	@param max_time (float):
		Time limit of the ISI and auto-correlogram (in ms).
	@param fr_bin_size (float):
		Size of bin for the firing rate histogram (in s).
	@param ms_before, ms_after (float):
		Timings for the waveform extraction (in ms).
	@param filename (str):
		Name of exported file. "units" by default.
	"""

	if len(unit_ids) == 0:
		return

	ISIs = []
	auto_corrs = []
	firing_rates = []
	all_wvfs = []
	all_channels = []

	for unit_id in unit_ids:
		ISI, ISI_bins = get_ISI(data, unit_id, bin_size, max_time)
		auto_corr, auto_corr_bins = get_autocorr(data, unit_id, bin_size, max_time)
		firing_rate, fr_bins = get_firing_rate(data, unit_id, fr_bin_size, as_Hz=True)

		ISIs.append(ISI)
		auto_corrs.append(auto_corr)
		firing_rates.append(firing_rate)

		waveforms = data.get_unit_waveforms(unit_id, ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=5000, max_channels_per_waveforms=3, return_idx=True)
		channels = waveforms[1]
		waveforms = np.mean(waveforms[0], axis=0, dtype=np.float32)
		wvf_xaxis = np.linspace(-ms_before, ms_after, waveforms.shape[1])

		all_wvfs.append(waveforms)
		all_channels.append(channels)
	
	ISIs = np.array(ISIs, dtype=np.uint32)
	auto_corrs = np.array(auto_corrs, dtype=np.uint32)
	firing_rates = np.array(firing_rates, dtype=np.float32)
	all_wvfs = np.array(all_wvfs, dtype=np.float32)
	all_channels = np.array(all_channels, dtype=np.uint16)

	plot_units_correlograms(unit_ids, save_folder, (ISIs, ISI_bins), (auto_corrs, auto_corr_bins), (firing_rates, fr_bins), (wvf_xaxis, all_wvfs, all_channels, data.uvolt_ratio), filename)


def plot_units_from_spiketrain(data, spike_trains: list, unit_ids: list, save_folder: str, bin_size: float=1.0/3.0, max_time: float=50.0,
							  fr_bin_size: float=2.0, ms_before: float=2, ms_after: float=2, filename: str="units"):
	"""

	"""

	if len(unit_ids) == 0 or len(spike_trains) == 0:
		return

	ISIs = []
	auto_corrs = []
	firing_rates = []
	all_wvfs = []
	all_channels = []

	for spike_train in spike_trains:
		ISI, ISI_bins = get_ISI_from_spiketrain(spike_train, bin_size, max_time, data.sampling_f)
		auto_corr, auto_corr_bins = get_autocorr_from_spiketrain(spike_train, bin_size, max_time, data.sampling_f)
		firing_rate, fr_bins = get_firing_rate_from_spiketrain(spike_train, data.recording.get_num_frames(), fr_bin_size, as_Hz=True, sampling_f=data.sampling_f)

		ISIs.append(ISI)
		auto_corrs.append(auto_corr)
		firing_rates.append(firing_rate)

		waveforms = data.wvf_extractor.get_waveforms_from_spiketrain(spike_train, ms_before=ms_before, ms_after=ms_after, max_channels_per_waveforms=3, return_idx=True)
		channels = waveforms[1]
		waveforms = np.mean(waveforms[0], axis=0, dtype=np.float32)
		wvf_xaxis = np.linspace(-ms_before, ms_after, waveforms.shape[1])

		all_wvfs.append(waveforms)
		all_channels.append(channels)
	
	ISIs = np.array(ISIs, dtype=np.uint32)
	auto_corrs = np.array(auto_corrs, dtype=np.uint32)
	firing_rates = np.array(firing_rates, dtype=np.float32)
	all_wvfs = np.array(all_wvfs, dtype=np.float32)
	all_channels = np.array(all_channels, dtype=np.uint16)

	plot_units_correlograms(unit_ids, save_folder, (ISIs, ISI_bins), (auto_corrs, auto_corr_bins), (firing_rates, fr_bins), (wvf_xaxis, all_wvfs, all_channels, data.uvolt_ratio), filename)


def plot_units_correlograms(unit_ids: list, save_folder: str, ISI: tuple, auto_corr: tuple, firing_rate: tuple, waveforms: tuple, filename: str="units"):
	"""
	Plots the characteristics of multiple units in a given folder (as one file).

	@param unit_ids (list of int):
		List of units' ID.
	@param save_folder (str):
		Path to the folder where the plot will be saved.
	@param ISI (tuple):
		Bins and correlograms for the ISI.
	@param auto_corr (tuple):
		Bins and correlograms of the auto-correlogram.
	@param firing_rate (tuple):
		Bins and correlograms of the firing rate (in Hz).
	@param waveforms (tuple):
		x-axis: np.ndarray (in ms).
		waveforms: np.ndarray [n_units, n_channels>=3, time].
		channel_idx: list [n_units, n_channels >= 3].
		uvolt_ratio: float or None.
	"""

	if len(unit_ids) == 0:
		return

	os.makedirs(save_folder, exist_ok=True)

	fig = make_subplots(rows=3, cols=2, subplot_titles=("ISI", "Mean waveform", "Auto-correlogram", "", "Firing rate", ""))
	steps = []

	fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
	fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
	fig.update_xaxes(title_text="Time (s)", row=3, col=1)
	fig.update_yaxes(title_text="Firing rate (Hz)", row=3, col=1)
	for i in range(3):
		fig.update_xaxes(title_text="Time (ms)", row=i+1, col=2)
		fig.update_yaxes(title_text="Voltage ({0})".format("A.U." if waveforms[3] is None else "µV"), row=i+1, col=2)

	for i in range(len(unit_ids)):
		unit_id = unit_ids[i]

		w = ISI[1][1] - ISI[1][0]
		fig.add_trace(go.Bar(
			x=ISI[1][:-1] + w/2,
			y=ISI[0][i],
			width=w,
			name="ISI",
			visible=False
		), row=1, col=1)

		w = auto_corr[1][1] - auto_corr[1][0]
		fig.add_trace(go.Bar(
			x=auto_corr[1][:-1] + w/2,
			y=auto_corr[0][i],
			width=w,
			name="Auto-correlogram",
			visible=False
		), row=2, col=1)

		w = firing_rate[1][1] - firing_rate[1][0]
		fig.add_trace(go.Bar(
			x=firing_rate[1][:-1] + w/2,
			y=firing_rate[0][i],
			width=w,
			name="Firing rate",
			visible=False
		), row=3, col=1)

		for j in range(3):
			fig.add_trace(go.Scatter(
				x=waveforms[0],
				y=waveforms[1][i, j] * (1 if waveforms[3] is None else waveforms[3]),
				mode="lines",
				name="Channel {0}".format(waveforms[2][i, j]),
				visible=False
			), row=j+1, col=2)

		step = dict(
			label="{0}".format(unit_id),
			method="update",
			args=[
				{"visible": [j//6 == i for j in range(6*len(unit_ids))]},
				{"title.text": "Unit {0}".format(unit_id),
				"annotations[4].text": "Frequency = {0:.2f} Hz".format(np.mean(firing_rate[0][i]))}
			]
		)
		steps.append(step)

	for i in range(6):
		fig.data[i].visible = True
	sliders = [dict(
		active=0,
		currentvalue={"prefix": "Unit "},
		pad={"t": 50},
		steps=steps
	)]
	
	fig.add_annotation(
		x=0.3,
		y=1.1,
		xref="paper",
		yref="paper",
		showarrow=False,
		text="Frequency = {0:.2f} Hz".format(np.mean(firing_rate[0][0]))
	)

	fig.update_layout(width=1440, height=900, sliders=sliders)
	fig.write_html("{0}/{1}.html".format(save_folder, filename))
