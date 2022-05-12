import os
import time
import math
import itertools
import numpy as np
import scipy.signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phy_data import PhyData
import postprocessing.center_waveform as center_waveform
import postprocessing.filter as filter
import postprocessing.utils as utils


def automerge_units(data: PhyData, unit_ids: list, params: dict, plot_folder: str):
	"""
	Uses a custom algorithm to find and check potential merges between units.
	If it passes all of the checks (correlograms, waveforms and score), then it merges them.

	@param data:
		The data object.
	@param unit_ids (list):
		List of units' id to potentialy merge together.
	@param params (dict):
		Parameters for the merge_units function.
	@param plot_folder (str):
		Path to the plot folder.

	@return new_units_id (dict):
		TODO
	"""

	kwargs = {
		"unit_ids": unit_ids,
		"refractory_period": [0.2, 1.0],
		"max_reshift": 0.5,
		"correlogram_check": {
			"max_time": 20.0,
			"bin_size": 1.0/3.0,
			"filter": [2, 800],
			"similarity": 0.7,
			"window": {
				"type": "adaptative",
				"limit": 10.0,
				"filter": [2, 300],
				"threshold_mean": 0.5
			}
		},
		"waveform_validation": {
			"similarity": 0.7,
			"filter": [2, 300, 6000],
			"waveforms": {
				"ms_before": 1.0,
				"ms_after": 1.0,
				"max_spikes_per_unit": 10000,
				"max_channels_per_waveforms": 5
			},
			"n_process": 1
		},
		"score_validation": {
			"good_noise_ratio": 1.5,
			"duplicates_window": 0.2
		},
		"plot_similarity": False
	}
	kwargs.update(params)

	new_units_id = dict()
	shifts = _get_cross_cluster_shift(data, unit_ids, kwargs)

	shifts2 = (shifts / (1e-3 * data.sampling_f * kwargs['correlogram_check']['bin_size'])).round().astype(np.int16)
	potential_merges = _get_potential_merges(data, unit_ids, shifts2, kwargs['correlogram_check'], plot_folder)

	shifts2 = np.zeros([len(potential_merges)], dtype=np.int16)
	for i in range(len(shifts2)):
		idx0 = np.where(unit_ids == potential_merges[i, 0])[0][0]
		idx1 = np.where(unit_ids == potential_merges[i, 1])[0][0]
		shifts2[i] = shifts[idx0, idx1]
	potential_merges = _waveform_validation(data, potential_merges, shifts2, kwargs['max_reshift'], kwargs['waveform_validation'], plot_folder)
	new_potential_merges = _score_validation(data, potential_merges, tuple(kwargs['refractory_period']), kwargs['score_validation'], plot_folder)

	k = 1 + kwargs['score_validation']['good_noise_ratio']
	for potential_merge in potential_merges:
		if potential_merge in new_potential_merges:
			continue

		f1 = data.get_unit_firing_rate(potential_merge[0])
		f2 = data.get_unit_firing_rate(potential_merge[1])
		C1 = utils.estimate_unit_contamination(data, potential_merge[0], tuple(kwargs['refractory_period']))
		C2 = utils.estimate_unit_contamination(data, potential_merge[1], tuple(kwargs['refractory_period']))
		score_1 = f1 * (1 - k*C1)
		score_2 = f2 * (1 - k*C2)

		unit_to_delete = potential_merge[1] if score_1 >= score_2 else potential_merge[0]
		new_units_id[unit_to_delete] = -1

	potential_merges = new_potential_merges

	shifts2 = np.zeros([len(potential_merges)], dtype=np.int16)
	for i in range(len(shifts2)):
		idx0 = np.where(unit_ids == potential_merges[i, 0])[0][0]
		idx1 = np.where(unit_ids == potential_merges[i, 1])[0][0]
		shifts2[i] = shifts[idx0, idx1]
	_plot_result(data, potential_merges, shifts2, kwargs, plot_folder)
	if kwargs['plot_similarity']:
		_plot_similarity(data, unit_ids, shifts, kwargs, plot_folder)

	merges = format_merges(potential_merges)
	merges = _recheck_score(data, merges, tuple(kwargs['refractory_period']), kwargs['score_validation'])

	for merge in merges:
		if len(merge) <= 1:
			continue

		for unit in merge[1:]:
			idx0 = np.where(unit_ids == merge[0])[0][0]
			idx1 = np.where(unit_ids == unit)[0][0]
			shift = shifts[idx0, idx1]
			data.change_spikes_time(unit, -shift)
		new_unit_id = data.sorting.merge_units(merge)

		for unit_id in merge:
			new_units_id[unit_id] = new_unit_id

	for unit_id in new_units_id.keys():
		if new_units_id[unit_id] == -1:
			data.sorting.exclude_units(unit_id)

	return new_units_id


def _get_potential_merges(data: PhyData, unit_ids: list, shifts: np.ndarray, params: dict, plot_folder: str, return_differences: bool=False):
	"""
	Returns a array of potential merges based on the correlograms.
	Compares the cross-correlogram to the auto-correlograms of 2 units,
	and if they are similar enough, they get labeled as potential merges.

	@param data (PhyData):
		Data object.
	@param unit_ids (list) [n_units]:
		List of units to check for merges (will check every units with every others).
	@param params (dict):
		Parameters for the merge_units function.
	@param plot_folder (str):
		Path to the plot folder.

	@return potential_merges (np.ndarray[uint16]) [n_potential_merges, 2]:
		List of all potential merges (by unit id).
	"""

	correlograms = data.get_all_correlograms(unit_ids, params['bin_size'], params['max_time'])[0].astype(np.int32)

	potential_merges = []
	m = correlograms.shape[2] // 2	# Index of the middle of the correlograms.
	N = np.array([len(data.sorting.get_unit_spike_train(unit_id)) for unit_id in unit_ids], dtype=np.uint32)	# Number of spikes in each unit.
	W = get_units_window(np.einsum('iij->ij', correlograms), params['bin_size'], params['window'])	# Window of each unit (in sampling time).

	b, a = filter.get_filter_params(params['filter'][0], max_f=params['filter'][1], sampling_rate=1e3/params['bin_size'], btype="lowpass")
	for i in range(len(unit_ids)):
		for j in range(len(unit_ids)):
			if i == j:
				continue
			correlograms[i, j, m-1:m+1] = 0
	correlograms = filter.filter(correlograms, b, a, np.float32)

	pairs_to_plot = []
	differences = []
	all_differences = []
	windows = []

	for i in range(len(unit_ids)):
		for j in range(i+1, len(unit_ids)):
			shift = shifts[i][j]
			window = int(round((N[i]*W[i] + N[j]*W[j]) / (N[i]+N[j])))	# Weighted window (larger unit imposes its window).
			plage = np.arange(m-window, m+window).astype(np.uint16)		# Plage of indices where correlograms are inside the window.

			diff1, diff2 = get_correlogram_differences(correlograms[i, i], correlograms[j, j], correlograms[i, j], plage, shift)
			diff = (N[i]*diff1 + N[j]*diff2) / (N[i]+N[j])	# Weighted difference (larger unit imposes its difference).
			all_differences.append(diff)

			if diff < 1 - params['similarity']:
				potential_merges.append([unit_ids[i], unit_ids[j]])
			
			if diff < 1.1 - params['similarity']:
				pairs_to_plot.append([i, j])
				differences.append(diff)
				windows.append(window)

	pairs_to_plot = np.array(pairs_to_plot, dtype=np.uint16)
	differences = np.array(differences, dtype=np.float32)
	windows = np.array(windows, dtype=np.float32) * params['bin_size']

	if plot_folder != None:
		_plot_correlogram_checks(correlograms, unit_ids, pairs_to_plot, differences, differences < 1-params['similarity'], params['max_time']-params['bin_size']/2, windows, plot_folder)

	if return_differences:
		return np.array(all_differences, dtype=np.float32)
	else:
		return np.array(potential_merges, dtype=np.uint32)


def get_correlogram_differences(auto_corr1: np.ndarray, auto_corr2: np.ndarray, cross_corr: np.ndarray, plage: np.ndarray, shift: int=0):
	"""
	Returns the differences between the auto_corrs on one side and the cross_corr on the other.
	First normalizes the correlogram so that the result doesn't depend on the height of the correlograms.
	WARNING: Do not send the correlograms as a uint, or else the difference will fail if negative.

	@param auto_corr1 (np.ndarray[int]) [time]:
		Auto-correlogram of the first unit.
	@param auto_corr2 (np.ndarray[int]) [time]:
		Auto-correlogram of the second unit.
	@param cross_corr (np.ndarray[int]) [time]:
		Cross-correlogram between the two units.

	@return diff1 (float):
		How different auto_corr1 is from cross_corr (0 means identical).
	@return diff2 (float):
		How different auto_corr2 is from cross_corr (0 means identical).
	"""

	auto_corr1 = normalize_correlogram(auto_corr1)
	auto_corr2 = normalize_correlogram(auto_corr2)
	cross_corr = normalize_correlogram(cross_corr)

	diff1 = np.sum(np.abs(cross_corr[plage-shift] - auto_corr1[plage])) / len(plage)
	diff2 = np.sum(np.abs(cross_corr[plage-shift] - auto_corr2[plage])) / len(plage)

	return (diff1, diff2)


def get_units_window(auto_corrs: np.ndarray, bin_size: float, params: dict):
	"""
	Returns the correlogram window for each unit based on the params.

	@param auto_corrs (np.,ndarray) [n_units, time]:
		Correlograms used to compute the adaptive window.
	@param bin_size (float):
		Size of a bin (in ms).
	@param params (dict):
		Parameters of merge_units.correlogram_window.

	@return units_window (np.ndarray[int16]) [n_units]:
		The window for each unit (as number of bins from center of correlogram).
	"""

	if params['type'] == "fix":
		return np.array([params['limit']/bin_size]*len(auto_corrs)).round().astype(np.int16)
	if params['type'] == "adaptative":
		return _get_units_adaptive_window(auto_corrs, bin_size, params)

	print("Incorrect type for correlogram_window: '{0}'".format(params['type']))
	assert False


def _get_units_adaptive_window(auto_corrs: np.ndarray, bin_size: float, params: dict):
	"""
	Runs _get_unit_adaptive_window for all correlograms.
	Low-pass filters the correlograms first before runnin this function.

	@param auto_corrs (np.ndarray) [n_correlograms, time]:
		Correlograms used for adaptive window. Needs to be from t=-max to +max.
	@param bin_size (float):
		Size of bin (in ms).
	@param params (dict):
		Parameters of merge_units.correlogram_window.

	@return units_window (np.ndarray[int16]) [n_correlograms]:
		Indices at which the adaptive window has been calculated for each correlogram.
	"""

	thresholds = np.max(auto_corrs, axis=1) * params['threshold_mean']
	limit = int(math.ceil(params['limit']/bin_size))

	b, a = filter.get_filter_params(order=params['filter'][0], max_f=params['filter'][1], sampling_rate=1e3/bin_size, btype="lowpass")
	filtered_autocorrs = filter.filter(auto_corrs, b, a, dtype=np.float32)[:, auto_corrs.shape[1]//2:auto_corrs.shape[1]//2+limit]

	windows = np.zeros([len(auto_corrs)], dtype=np.int16)
	for i in range(len(auto_corrs)):
		windows[i] = _get_unit_adaptive_window(filtered_autocorrs[i], threshold=thresholds[i])

	return windows


def _get_unit_adaptive_window(auto_corr: np.ndarray, threshold: float):
	"""
	Computes an adaptive window to correlogram (basically corresponds to the first peak).
	Based on a minimum threshold and minimum of second derivative.
	If no peak is found over threshold, recomputes with threshold/2.

	@param auto_corr (np.ndarray) [time]:
		Correlogram used for adaptive window. Needs to start at t=0.
	@param threshold (float):
		Minimum threshold of correlogram (all peaks under this threshold is discarded).

	@return unit_window (int):
		Index at which the adaptive window has been calculated.
	"""

	if np.sum(np.abs(auto_corr)) == 0:
		return 20.0

	peaks = scipy.signal.find_peaks(-np.gradient(np.gradient(auto_corr)))[0]

	for peak in peaks:
		if auto_corr[peak] >= threshold:
			return peak

	# If none of the peaks crossed the threshold, redo with threshold/2.
	return _get_unit_adaptive_window(auto_corr, threshold/2)


def _waveform_validation(data: PhyData, potential_merges: np.ndarray, shifts: np.ndarray, max_shift: float, params: dict, plot_folder: str, return_differences: bool=False):
	"""
	Computes a waveform validation on all potential merges.
	Looks at the mean of the units' filtered waveforms, and check if they are similar enough.

	@param data (PhyData):
		Data object.
	@param potential_merges (np.ndarray) [n_potential_merges, 2]:
		List containing the units' id of potential merges.
	@param params (dict):
		Parameters of merge_units.waveform_validation.
	@param plot_folder (str):
		Path to the plot folder.

	@return potential_merges (np.ndarray) [n'_potential_merges, 2]:
		List of potential merges that passed the waveform validation.
	"""

	validated_merges = []
	differences = []
	unit_ids = np.unique(potential_merges.flatten())
	max_shift_samples = int(max_shift * data.sampling_f * 1e-3)

	b, a = filter.get_filter_params(params['filter'][0], params['filter'][1], params['filter'][2], btype="bandpass")
	waveforms = data.get_units_mean_waveform(unit_ids, ms_before=params['waveforms']['ms_before']+max_shift, ms_after=params['waveforms']['ms_after']+max_shift, max_spikes_per_unit=params['waveforms']['max_spikes_per_unit'], max_channels_per_waveforms=None)
	waveforms = np.array(filter.filter_units_waveforms_parallel(waveforms, b, a, dtype=np.float32, n_process=params['n_process']), dtype=np.float32)

	for i in range(len(potential_merges)):
		shift = shifts[i]
		wfs_1 = waveforms[np.argmax(unit_ids == potential_merges[i, 0])][:, max_shift_samples : -max_shift_samples]
		wfs_2 = waveforms[np.argmax(unit_ids == potential_merges[i, 1])][:, max_shift_samples-shift : waveforms.shape[2]-max_shift_samples-shift]

		difference = compute_waveforms_difference(wfs_1, wfs_2, n_max_channels=params['waveforms']['max_channels_per_waveforms'], max_shift=0)
		differences.append(difference)
		if difference < 1 - params['similarity']:
			validated_merges.append(i)

	differences = np.array(differences, dtype=np.float32)

	if plot_folder != None:
		_plot_waveform_checks(waveforms, potential_merges, shifts, max_shift_samples, differences, differences < 1-params['similarity'], params['waveforms']['max_channels_per_waveforms'], (params['waveforms']['ms_before'], params['waveforms']['ms_after']), data.uvolt_ratio, plot_folder)

	if return_differences:
		return differences
	else:
		return potential_merges[validated_merges]


def compute_waveforms_difference(waveform_1: np.ndarray, waveform_2: np.ndarray, n_max_channels=None, max_shift: int=0, return_channels: bool=False):
	"""
	Computes a score telling how different two waveforms are (0 being identical).

	@param waveform_1 (np.ndarray) [n_channels, time]:
		First waveform to compare.
	@param waveform_2 (np.ndarray) [n_channels, time]:
		Second waveform to compare to the first one.
	@param n_max_channels (int or None):
		Maximum number of channels to compare.
		If n_channels > n_max_channels, will extract the best n_max_channels channels.

	@return waveforms_difference (float >= 0):
		How different the two waveforms are (0 being identical).
	"""

	if isinstance(n_max_channels, int) and n_max_channels < waveform_1.shape[0]:
		best_channels = np.argsort(np.max(np.abs(waveform_1+waveform_2), axis=1))[:-n_max_channels-1:-1]
		waveform_1 = waveform_1[best_channels]
		waveform_2 = waveform_2[best_channels]

	score = 5
	for shift in range(-max_shift, max_shift+1):
		s = np.sum(np.abs(waveform_1[:, max_shift:waveform_1.shape[1]-max_shift] - waveform_2[:, max_shift+shift:waveform_2.shape[1]-max_shift+shift])) / (np.sum(np.abs(waveform_1)) + np.sum(np.abs(waveform_2)))
		if s < score:
			score = s

	if return_channels:
		return (score, best_channels)
	else:
		return score


def _score_validation(data: PhyData, potential_merges: np.ndarray, refractory_period: tuple, params: dict, plot_folder: str):
	"""
	Validates a merge or not based on the score of the merged unit vs score of individual units.
	This to avoid merging a very contaminated unit with a beautiful one.

	@param data (PhyData):
		The data object.
	@param potential_merges (np.ndarray) [n_potential_merges, 2]:
		List containing the units' id of potential merges.
	@param refractory_period (tuple of 2 float):
		Window of refractory period to estimate contamination (in ms).
	@param params (dict):
		Parameters of merge_units.score_validation.

	@return potential_merges (np.ndarray) [n'_potential_merges, 2]:
		List of potential merges that passed the score validation.
	"""

	if len(potential_merges) == 0:
		return potential_merges

	k = 1 + params['good_noise_ratio']
	validated_merges = []

	logs = open("{0}/score_validation.logs".format(plot_folder), 'x')

	for i in range(len(potential_merges)):
		unit1, unit2 = potential_merges[i]
		logs.write("Units {0}-{1}:\n".format(unit1, unit2))

		merged_spike_train = np.sort(np.concatenate((data.get_unit_spike_train(unit1), data.get_unit_spike_train(unit2))))
		duplicates = np.argwhere(np.diff(merged_spike_train) <= params['duplicates_window'] * 1e-3 * data.sampling_f)[:, 0]
		merged_spike_train = np.delete(merged_spike_train, duplicates)

		f1 = data.get_unit_firing_rate(unit1)
		f2 = data.get_unit_firing_rate(unit2)
		fm = len(merged_spike_train) / data.recording.get_num_frames() * data.sampling_f
		C1 = utils.estimate_unit_contamination(data, unit1, refractory_period)
		C2 = utils.estimate_unit_contamination(data, unit2, refractory_period)
		Cm = utils.estimate_spike_train_contamination(merged_spike_train, refractory_period, data.recording.get_num_frames(), data.sampling_f)

		score_1 = f1 * (1 - k*C1)
		score_2 = f2 * (1 - k*C2)
		score_m = fm * (1 - k*Cm)
		to_merge = score_m >= score_1 and score_m >= score_2

		logs.write("\t- Unit {0}:\n".format(unit1))
		logs.write("\t\t* Freq = {0:.2f} Hz\n".format(f1))
		logs.write("\t\t* Cont = {0:.1f} %\n".format(100*C1))
		logs.write("\t\t* Score: {0:.2f}\n".format(score_1))
		logs.write("\t- Unit {0}:\n".format(unit2))
		logs.write("\t\t* Freq = {0:.2f} Hz\n".format(f2))
		logs.write("\t\t* Cont = {0:.1f} %\n".format(100*C2))
		logs.write("\t\t* Score: {0:.2f}\n".format(score_2))
		logs.write("\t- Merged unit:\n")
		logs.write("\t\t* Freq = {0:.2f} Hz\n".format(fm))
		logs.write("\t\t* Cont = {0:.1f} %\n".format(100*Cm))
		logs.write("\t\t* Score: {0:.2f}\n".format(score_m))
		logs.write("\t=> Merge {0}.\n\n".format("accepted" if to_merge else "denied"))

		if to_merge:
			validated_merges.append(i)

	logs.close()
	return potential_merges[validated_merges]


def _recheck_score(data: PhyData, merges: list, refractory_period: tuple, params: dict):
	"""
	Rechecks the merging score on units that have been sorted by same neuron.
	Returns only the units that will give the best score on merge.

	@param data (PhyData):
		The data object.
	@param merges (list):
		List of all units that belong to the same neuron.
	@param refractory_period (tuple of 2 float):
		Window of refractory period to estimate contamination (in ms).
	@param params (dict):
		Parameters of merge_units.score_validation.

	@return merges (list) [n_merges][n_units]:
		List of all definitive merges.
	"""

	k = 1 + params['good_noise_ratio']

	for i in range(len(merges)):
		merge = merges[i]

		score = -4000
		units = []

		for n_units in range(1, 1 + min(4, len(merge))):
			for combination in itertools.combinations(merge, n_units):
				F = sum([data.get_unit_firing_rate(unit_id) for unit_id in combination])
				C = utils.estimate_units_contamination(data, combination, refractory_period)
				S = F * (1 - k*C)

				if S > score:
					score = S
					units = combination

		if len(merge) >= 4:
			for n_units in range(len(merge)-2, len(merge)):
				for combination in itertools.combinations(merge, n_units):
					F = sum([data.get_unit_firing_rate(unit_id) for unit_id in combination])
					C = utils.estimate_units_contamination(data, combination, refractory_period)
					S = F * (1 - k*C)

					if S > score:
						score = S
						units = combination

		merges[i] = list(units)

	return merges


def normalize_correlogram(correlogram: np.ndarray):
	"""
	Normalizes a correlogram so its mean in time is 1.
	If correlogram is 0 everywhere, stays 0 everywhere.

	@param correlogram (np.ndarray) [time]:
		Correlogram to normalize.

	@return normalized_correlogram (np.ndarray) [time]:
		Normalized correlogram to have a mean of 1.
	"""

	mean = np.mean(correlogram)
	return correlogram if mean == 0 else correlogram/mean


def format_merges(pairs: np.ndarray):
	"""
	Format an array of pair of merges into a list of all chains of the same unit.
	For exemple, [[1, 8], [2, 4], [3, 5], [2, 8]] becomes [[1, 8, 2, 4], [3, 5]].

	@param pairs (np.ndarray[int]) [n_pairs, 2]:
		Given pairs of merges.

	@return merges (list of list of int):
		Final merges per unit.
	"""

	merges = []

	for pair in pairs:
		if any(pair[0] in m for m in merges):
			continue

		merges.append(_find_all_connections(pair[0], pairs, []))

	return merges


def _find_all_connections(unit: int, pairs: np.ndarray, conn: list):
	"""
	Uses recursion to find all connections step by step of 'unit' in 'pairs' and adds it to 'conn'.

	@param unit (int):
		Integer to look for connections.
	@param pairs (np.ndarray[int]) [n_pairs, 2]:
		All pairs of merges.
	@param conn (list):
		Given list of units already connecter with 'unit'.
		Give [] to leave the function to find all connections.

	@return conn (list):
		Updated conn with all connections step by step.
	"""

	if unit in conn:
		return conn

	conn.append(unit)

	for pair in pairs:
		if unit not in pair:
			continue

		other_unit = np.sum(pair) - unit
		if other_unit in conn:
			continue

		conn = _find_all_connections(other_unit, pairs, conn)

	return conn


def _get_cross_cluster_shift(data: PhyData, unit_ids: list, params: dict):
	"""

	"""

	shifts = np.zeros([len(unit_ids), len(unit_ids)], dtype=np.int16)
	max_shift = params['max_reshift']
	max_shift_samples = int(max_shift * data.sampling_f * 1e-3)

	b, a = filter.get_filter_params(params['waveform_validation']['filter'][0], params['waveform_validation']['filter'][1], params['waveform_validation']['filter'][2], btype="bandpass")
	mean_waveforms = data.get_units_mean_waveform(unit_ids, ms_before=params['waveform_validation']['waveforms']['ms_before'] + max_shift,
						ms_after=params['waveform_validation']['waveforms']['ms_after'] + max_shift, max_spikes_per_unit=params['waveform_validation']['waveforms']['max_spikes_per_unit'],
						max_channels_per_waveforms=None)

	mean_waveforms = filter.filter(mean_waveforms, b, a, dtype=np.float32)

	for i in range(len(unit_ids)):
		conv = center_waveform.compute_convolution(mean_waveforms, mean_waveforms[i, :, max_shift_samples:-max_shift_samples])
		shifts[i] = np.argmax(conv, axis=1) - max_shift_samples

	return -shifts




def _plot_correlogram_checks(correlograms: np.ndarray, unit_ids: list, pairs: list, diff: np.ndarray, passed: list, max_time: float, windows: list, plot_folder: str):
	"""
	Plots the result of the _get_potential_merges function.

	@param correlograms (np.ndarray) [n_units, n_units, time]:
		Auto- and cross-correlograms of all the units.
	@param unit_ds (list or np.ndarray[int]) [n_units]:
		The ID of all units units.
	@param pairs (list of np.ndarray[int]) [n_pairs, 2]:
		The index (not ID!) of units by pair.
	@param diff (np.ndarray) []:
		The computed difference between the two correlograms.
	@param passed (list of bool) [n_pairs]:
		Did the correlogram check pass (i.e is it a potential unit?).
	@param max_time (float):
		The time limit on the correlograms (in ms).
	@param windows (list of float) [n_pairs]:
		The time limit on which the check was computed (in ms).
	@param plot_folder (str):
		Path to the plot folder.
	"""

	if len(unit_ids) == 0 or len(correlograms) == 0 or len(pairs) == 0:
		return

	os.makedirs(plot_folder, exist_ok=True)

	fig = make_subplots(rows=1, cols=2, shared_xaxes=True,
						subplot_titles=("Auto-correlograms", "Cross-correlogram"))
	steps = []
	xaxis = np.linspace(-max_time, max_time, correlograms.shape[2])

	fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
	fig.update_xaxes(title_text="Time (ms)", row=1, col=2)

	for i in range(len(pairs)):
		pair = pairs[i]

		fig.add_trace(go.Scatter(
			x=xaxis,
			y=correlograms[pair[0], pair[0]],
			mode="lines",
			name="Unit {0}".format(unit_ids[pair[0]]),
			marker_color="CornflowerBlue",
			visible=False
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis,
			y=correlograms[pair[1], pair[1]],
			mode="lines",
			name="Unit {0}".format(unit_ids[pair[1]]),
			marker_color="LightSeaGreen",
			visible=False
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis,
			y=correlograms[pair[0], pair[1]],
			mode="lines",
			name="Units {0}-{1}".format(*unit_ids[pair]),
			marker_color="Crimson",
			visible=False
		), row=1, col=2)

		fig.add_trace(go.Scatter(x=[-windows[i]]*2, y=[0, max(np.max(correlograms[pair[0], pair[0]]), np.max(correlograms[pair[1], pair[1]]))], mode="lines", marker_color="red", name="Check window", legendgroup="window", visible=False), row=1, col=1)
		fig.add_trace(go.Scatter(x=[+windows[i]]*2, y=[0, max(np.max(correlograms[pair[0], pair[0]]), np.max(correlograms[pair[1], pair[1]]))], mode="lines", marker_color="red", name="Check window", showlegend=False, legendgroup="window", visible=False), row=1, col=1)
		fig.add_trace(go.Scatter(x=[-windows[i]]*2, y=[0, np.max(correlograms[pair[0], pair[1]])], mode="lines", marker_color="red", name="Check window", showlegend=False, legendgroup="window", visible=False), row=1, col=2)
		fig.add_trace(go.Scatter(x=[+windows[i]]*2, y=[0, np.max(correlograms[pair[0], pair[1]])], mode="lines", marker_color="red", name="Check window", showlegend=False, legendgroup="window", visible=False), row=1, col=2)

		step = dict(
			label="{0}-{1}".format(*unit_ids[pair]),
			method="update",
			args=[
				{"visible": [j//7 == i for j in range(7*len(pairs))]},
				{"title.text": "Units {0}-{1} (Difference = {2:.2f})".format(*unit_ids[pair], diff[i]),
				"title.font.color": "black" if passed[i] else "red"}
			]
		)
		steps.append(step)

	for i in range(7):
		fig.data[i].visible = True
	sliders = [dict(
		active=0,
		currentvalue={"prefix": "Units "},
		pad={"t": 50},
		steps=steps
	)]

	fig.update_layout(width=1440, height=810, sliders=sliders, yaxis_rangemode="tozero")
	fig.write_html("{0}/correlogram_checks.html".format(plot_folder))


def _plot_waveform_checks(waveforms: np.ndarray, potential_merges: np.ndarray, shifts: np.ndarray, max_shift: int, scores: np.ndarray, passed: np.ndarray, n_max_channels: int, time_bound: tuple, uvolt_ratio, plot_folder: str):
	"""
	Plots the result of the _waveform_validation function.

	@param waveforms (np.ndarray) [n_units, n_channels, time]:
		Filtered mean waveform of all the units.
	@param potential_merges (np.ndarray[int]) [n_pairs, 2]:
		The ID of all units units by pairs.
	@param scores (np.ndarray[float]) [n_pairs]:
		The computed difference between the two mean waveforms.
	@param passed (np.ndarray[bool]) [n_pairs]:
		Did the waveform check pass (i.e is it a potential unit?).
	@param n_max_channels (int):
		Maximum number of channels to use for comparison.
	@param time_bound (tuple of 2 float):
		Time axis bound for plot (in ms).
	@param uvolt_ratio (float or None):
		Ratio to go to µV. None for arbitrary units.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	if len(waveforms) == 0 or len(potential_merges) == 0:
		return

	os.makedirs(plot_folder, exist_ok=True)

	unit_ids = np.unique(potential_merges.flatten())

	n_rows = n_max_channels // 5
	n_cols = math.ceil(n_max_channels / n_rows)
	fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, shared_yaxes=True)
	for row in range(1, n_rows+1):
		for col in range(1, n_cols+1):
			fig.update_xaxes(title_text="Time (ms)", row=row, col=col)
			fig.update_yaxes(title_text="Voltage ({0})".format("A.U." if uvolt_ratio is None else "µV"), row=row, col=col)

	steps = []
	xaxis = np.linspace(-time_bound[0], time_bound[1], waveforms.shape[2])

	for i in range(len(potential_merges)):
		shift = shifts[i]
		wfs_1 = waveforms[np.argmax(unit_ids == potential_merges[i, 0])][:, max_shift : -max_shift]
		wfs_2 = waveforms[np.argmax(unit_ids == potential_merges[i, 1])][:, max_shift-shift : waveforms.shape[2]-max_shift-shift]

		if isinstance(n_max_channels, int) and n_max_channels < wfs_1.shape[0]:
			best_channels = np.argsort(np.max(np.abs(wfs_1+wfs_2), axis=1))[:-n_max_channels-1:-1]
			wfs_1 = wfs_1[best_channels]
			wfs_2 = wfs_2[best_channels]

		for channel in range(n_max_channels):
			fig.add_trace(go.Scatter(
				x=xaxis,
				y=wfs_1[channel] * (1 if uvolt_ratio is None else uvolt_ratio),
				mode="lines",
				marker_color="CornflowerBlue",
				name="Unit {0} (channel {1})".format(potential_merges[i, 0], best_channels[channel]),
				visible=False
			), row=1+channel//n_cols, col=1+channel%n_cols)
			fig.add_trace(go.Scatter(
				x=xaxis,
				y=wfs_2[channel] * (1 if uvolt_ratio is None else uvolt_ratio),
				mode="lines",
				marker_color="LightSeaGreen",
				name="Unit {0} (channel {1})".format(potential_merges[i, 1], best_channels[channel]),
				visible=False
			), row=1+channel//n_cols, col=1+channel%n_cols)

		step = dict(
			label="{0}-{1}".format(*potential_merges[i]),
			method="update",
			args=[
				{"visible": [j//(2*n_max_channels) == i for j in range(2*n_max_channels*len(potential_merges))]},
				{"title.text": "Units {0}-{1} (Difference = {2:.2f})".format(*potential_merges[i], scores[i]),
				"title.font.color": "black" if passed[i] else "red"}
			]
		)
		steps.append(step)

	for i in range(2*n_max_channels):
		fig.data[i].visible = True
	sliders = [dict(
		active=0,
		currentvalue={"prefix": "Units "},
		pad={"t": 50},
		steps=steps
	)]

	fig.update_layout(width=1440, height=810, sliders=sliders)
	fig.write_html("{0}/waveform_checks.html".format(plot_folder))


def _plot_result(data: PhyData, potential_merges: np.ndarray, shifts: np.ndarray, params: dict, plot_folder: str):
	"""
	Plots the result of merging (for each merge pair, correlograms and waveforms).

	@param data (PhyData):
		The data object.
	@param potential_merges (np.ndarray) [n_merges, 2]:
		List of units' id pair to merge.
	@param correlograms (np.ndarray) [n_units, n_units, time]:
		The auto- and cross-correlograms between the units (only units in potential_merges).
	@param params (dict):
		Parameters for the merge_units function.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	if len(potential_merges) == 0:
		return

	os.makedirs(plot_folder, exist_ok=True)

	unit_ids = np.unique(potential_merges.flatten())

	correlograms, bins = data.get_all_correlograms(unit_ids, params['correlogram_check']['bin_size'], params['correlogram_check']['max_time'])
	waveforms = data.get_units_mean_waveform(unit_ids, ms_before=params['waveform_validation']['waveforms']['ms_before'], ms_after=params['waveform_validation']['waveforms']['ms_after'], max_spikes_per_unit=params['waveform_validation']['waveforms']['max_spikes_per_unit'], max_channels_per_waveforms=None)

	fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
						subplot_titles=("Auto-correlograms", "Mean waveform", "Cross-correlogram", ""))
	steps = []
	w = bins[1] - bins[0]
	xaxis_corr = np.linspace(bins[0]+w/2, bins[-1]-w/2, correlograms.shape[2])
	xaxis_wvfs = np.linspace(-params['waveform_validation']['waveforms']['ms_before'], params['waveform_validation']['waveforms']['ms_after'], waveforms.shape[2])

	for i in range(2):
		fig.update_xaxes(title_text="Time (ms)", row=i+1, col=1)
		fig.update_xaxes(title_text="Time (ms)", row=i+1, col=2)
		fig.update_yaxes(title_text="Voltage ({0})".format("A.U." if data.uvolt_ratio is None else "µV"), row=i+1, col=2)

	for i in range(len(potential_merges)):
		unit1 = potential_merges[i, 0]
		unit2 = potential_merges[i, 1]
		idx1 = np.argmax(unit_ids == unit1)
		idx2 = np.argmax(unit_ids == unit2)

		freq1 = data.get_unit_firing_rate(unit1)
		freq2 = data.get_unit_firing_rate(unit2)
		cont1 = utils.estimate_unit_contamination(data, unit1, tuple(params['refractory_period']))
		cont2 = utils.estimate_unit_contamination(data, unit2, tuple(params['refractory_period']))

		best_channels = np.argsort(np.max(np.abs(waveforms[idx1]+waveforms[idx2]), axis=1))[::-1]

		fig.add_trace(go.Scatter(
			x=xaxis_corr,
			y=correlograms[idx1, idx1],
			mode="lines",
			name="Unit {0} (autocorr)".format(unit1),
			marker_color="CornflowerBlue",
			visible=False
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis_corr,
			y=correlograms[idx2, idx2],
			mode="lines",
			name="Unit {0} (autocorr)".format(unit2),
			marker_color="LightSeaGreen",
			visible=False
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis_corr,
			y=correlograms[idx1, idx2],
			mode="lines",
			name="Units {0}-{1} (crosscorr)".format(unit1, unit2),
			marker_color="Crimson",
			visible=False
		), row=2, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis_wvfs,
			y=waveforms[idx1, best_channels[0]] * (1 if data.uvolt_ratio is None else data.uvolt_ratio),
			mode="lines",
			name="Unit {0} (channel {1})".format(unit1, best_channels[0]),
			marker_color="CornflowerBlue",
			visible=False
		), row=1, col=2)
		fig.add_trace(go.Scatter(
			x=xaxis_wvfs,
			y=waveforms[idx2, best_channels[0]] * (1 if data.uvolt_ratio is None else data.uvolt_ratio),
			mode="lines",
			name="Unit {0} (channel {1})".format(unit2, best_channels[0]),
			marker_color="LightSeaGreen",
			visible=False
		), row=1, col=2)
		fig.add_trace(go.Scatter(
			x=xaxis_wvfs,
			y=waveforms[idx1, best_channels[1]] * (1 if data.uvolt_ratio is None else data.uvolt_ratio),
			mode="lines",
			name="Unit {0} (channel {1})".format(unit1, best_channels[1]),
			marker_color="CornflowerBlue",
			visible=False
		), row=2, col=2)
		fig.add_trace(go.Scatter(
			x=xaxis_wvfs,
			y=waveforms[idx2, best_channels[1]] * (1 if data.uvolt_ratio is None else data.uvolt_ratio),
			mode="lines",
			name="Unit {0} (channel {1})".format(unit2, best_channels[1]),
			marker_color="LightSeaGreen",
			visible=False
		), row=2, col=2)

		step = dict(
			label="{0}-{1}".format(unit1, unit2),
			method="update",
			args=[
				{"visible": [j//7 == i for j in range(7*len(potential_merges))]},
				{"title.text": "Merge units {0}-{1}".format(unit1, unit2),
				"annotations": [
					dict(x=0.3, y=1.1, xref="paper", yref="paper", showarrow=False,
						text="Unit {0}\n- {1:.2f} Hz\n- {2:.2f} % cont".format(unit1, freq1, 100*cont1)),
					dict(x=0.6, y=1.1, xref="paper", yref="paper", showarrow=False,
						text="Unit {0}\n- {1:.2f} Hz\n- {2:.2f} % cont".format(unit2, freq2, 100*cont2)),
					dict(x=0.9, y=1.1, xref="paper", yref="paper", showarrow=False,
						text="Shift = {0} pt".format(shifts[i]))
				]}
			]
		)
		steps.append(step)

	for i in range(7):
		fig.data[i].visible = True
	sliders = [dict(
		active=0,
		currentvalue={"prefix": "Units "},
		pad={"t": 50},
		steps=steps
	)]

	fig.update_layout(width=1440, height=810, sliders=sliders, yaxis_rangemode="tozero")
	fig.write_html("{0}/results.html".format(plot_folder))


def _plot_similarity(data: PhyData, unit_ids: list, shifts: np.ndarray, params: dict, plot_folder: str):
	"""

	"""

	if len(unit_ids) == 0:
		return

	os.makedirs(plot_folder, exist_ok=True)

	all_pairs = np.array(list(itertools.combinations(unit_ids, 2)))
	shifts1 = (shifts / (1e-3 * data.sampling_f * params['correlogram_check']['bin_size'])).round().astype(np.int16)
	shifts2 = np.zeros([len(all_pairs)], dtype=np.int16)
	for i in range(len(all_pairs)):
		idx0 = np.where(unit_ids == all_pairs[i, 0])[0][0]
		idx1 = np.where(unit_ids == all_pairs[i, 1])[0][0]
		shifts2[i] = shifts[idx0, idx1]

	corr_diff = _get_potential_merges(data, unit_ids, shifts1, params['correlogram_check'], None, return_differences=True)
	wvf_diff = _waveform_validation(data, all_pairs, shifts2, params['max_reshift'], params['waveform_validation'], None, return_differences=True)

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=corr_diff,
		y=wvf_diff,
		mode="markers",
		name="Pairs difference",
		text=["Units {0}-{1}".format(*pair) for pair in all_pairs]
	))

	fig.add_shape(type="rect", x0=0, y0=0, x1=1-params['correlogram_check']['similarity'], y1=1-params['waveform_validation']['similarity'],
		line=dict(color="red", dash="dot"))

	fig.update_xaxes(title_text="Correlogram difference", range=[-0.05, 2.05])
	fig.update_yaxes(title_text="Waveform difference", range=[-0.03, 1.03])
	fig.update_layout(width=1440, height=900, title="Differences between pairs")
	fig.write_html("{0}/similarity.html".format(plot_folder))

	np.save("{0}/corr_diff.npy".format(plot_folder), corr_diff)
	np.save("{0}/wvf_diff.npy".format(plot_folder), wvf_diff)
