import os
import time
import math
import itertools
from multiprocessing import Pool
import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phy_data import PhyData
import postprocessing.center_waveform as center_waveform
import postprocessing.filter as filter
import postprocessing.merge_units as merge_units
import postprocessing.utils as utils


def automerge_sortings(data: PhyData, unit_ids: dict, params: dict, plot_folder: str):
	"""

	"""

	kwargs = {
		"refractory_period": [0.2, 1.0],
		"max_reshift": 0.5,
		"duplicated_spike": 0.2,
		"similarity_validation": {
			"window": 0.2,
			"min_similarity": 0.4,
			"n_process": 6
		},
		"correlogram_validation": {
			"bin_size": 5.0,
			"max_time": 200.0,
			"similarity": 0.7
		},
		"waveform_validation": {
			"similarity": 0.7,
			"filter": [2, 300, 6000],
			"waveforms": {
				"ms_before": 1.0,
				"ms_after": 1.0,
				"max_spikes_per_unit": 20000,
				"max_channels_per_waveforms": 5
			},
			"n_process": 6
		},
		"plot_units": {
			"ms_before": 2.0,
			"ms_after": 2.0,
			"max_time": 50.0,
			"bin_size": 1.0/3.0,
			"fr_bin_size": 2.0
		}
	}
	kwargs.update(params)
	kwargs['duplicated_spike'] = int(round(kwargs['duplicated_spike'] * 1e-3 * data.sampling_f))
	kwargs['similarity_validation']['window'] = int(round(kwargs['similarity_validation']['window'] * 1e-3 * data.sampling_f))

	if len(unit_ids) == 0:
		return

	#_plot_all_units(data, unit_ids, kwargs['plot_units'], plot_folder)
	os.makedirs(plot_folder, exist_ok=True)
	
	shifts = _get_cross_cluster_shift(data, unit_ids, kwargs)
	similar_units = get_similar_units(data, unit_ids, shifts, kwargs['similarity_validation'], plot_folder)
	similar_units = _correlogram_validation(data, similar_units, kwargs['correlogram_validation'], plot_folder)

	shifts2 = np.zeros([len(similar_units)], dtype=np.int16)
	for i in range(len(shifts2)):
		sorting1, unit1 = similar_units[i, 0]
		sorting2, unit2 = similar_units[i, 1]
		idx0 = np.where(unit_ids[sorting1] == unit1)[0][0]
		idx1 = np.where(unit_ids[sorting2] == unit2)[0][0]
		shifts2[i] = shifts[sorting1][sorting2][idx0, idx1]

	similar_units = _waveform_validation(data, similar_units, shifts2, kwargs['max_reshift'], kwargs['waveform_validation'], plot_folder)

	if len(similar_units) == 0:
		return

	similar_units = 10000*similar_units[:, :, 0] + similar_units[:, :, 1]
	similar_units = merge_units.format_merges(similar_units)
	similar_units = [[np.array([unit//10000, unit%10000], dtype=np.uint32) for unit in cluster] for cluster in similar_units]

	starting_ID = data.merged_sorting.get_unit_ids()[-1] + 1 if len(data.merged_sorting.get_unit_ids()) > 0 else 0
	shifts2 = []
	for i in range(len(similar_units)):
		shifts2.append([])

		for j in range(len(similar_units[i])):
			if j==0:
				shifts2[-1].append(0)
				continue

			sorting1, unit1 = similar_units[i][0]
			sorting2, unit2 = similar_units[i][j]
			idx0 = np.where(unit_ids[sorting1] == unit1)[0][0]
			idx1 = np.where(unit_ids[sorting2] == unit2)[0][0]
			shifts2[-1].append(shifts[sorting1][sorting2][idx0, idx1])

	perform_final_merging(data, similar_units, shifts2, kwargs, plot_folder)
	check_conflicts(data, starting_ID, kwargs, plot_folder)

	if len(unit_ids) <= 3:
		add_leftover_units(data, unit_ids, kwargs, plot_folder) # A cluster that is not detected in other analyses is usually not a neuron ; see Buccino & Hurwitz et Al. 2020

	data.clear_wvfs()


def get_similar_units(data: PhyData, unit_ids: dict, shifts: dict, params: dict, plot_folder: str):
	"""
	Computes the similar units based on coincident spikes.

	@params data (PhyData):
		The data object.
	@params unit_ids (dict):
		Dictionary containing all of the unit_ids (values) for all sortings (keys).
	@params shifts (dict):
		shifts[i][j] contains a 2D array with all of the shifts between the clusters of sortings i and j.
	@params params (dict):
		The parameters for the similarity validation step.
	@params plot_folder (str):
		Path to the logs folder.
	"""

	logs = open("{0}/similar_units.logs".format(plot_folder), 'x')
	agreement_matrices = _compute_agreement_matrices_parallel(data._sortings, unit_ids, shifts, params['window'], params['n_process'])

	similar_units = []
	for sorting1, d in agreement_matrices.items():
		for sorting2, agreement_matrix in d.items():
			units = np.array(np.where(np.max(agreement_matrix, axis=2) >= params['min_similarity']), dtype=np.uint16).T

			for couple in units:
				unit_id_1 = unit_ids[sorting1][couple[0]]
				unit_id_2 = unit_ids[sorting2][couple[1]]
				logs.write("Units {0}-{1} & {2}-{3}\n".format(sorting1, unit_id_1, sorting2, unit_id_2))
				similar_units.append([[sorting1, unit_id_1], [sorting2, unit_id_2]])

	logs.close()
	return np.array(similar_units, dtype=np.uint32)


def _compute_agreement_matrices(sortings: list, unit_ids: dict, shifts: dict, window: int):
	"""
	Computes the agreement matrices (i.e. the similarity in spike train between each pair of clusters between each pair of sortings).

	@params sortings (list):
		All the sortings.
	@params unit_ids (dict):
		Dictionnary with key=index of sorting in sortings list ; and values = list of unit ids to compute.
	@params shifts (dict):
		shifts[i][j] contains a 2D array with all of the shifts between the clusters of sortings i and j.
	@param window (int):
		The number of time intervals to consider 2 spikes coincident.
	"""

	agreement_matrices = dict()

	for i in list(unit_ids.keys())[:-1]:
		agreement_matrices[i] = dict()

		for j in unit_ids.keys():
			if j <= i:
				continue

			roots1 = [root.get_spike_train().astype(np.uint64) for root in sortings[i]._roots if root in unit_ids[i]]
			roots2 = [root.get_spike_train().astype(np.uint64) for root in sortings[j]._roots if root in unit_ids[j]]
			agreement_matrices[i][j] = _compute_agreement_matrix(roots1, roots2, shifts[i][j], window)

	return agreement_matrices


def _compute_agreement_matrix(roots1, roots2, shifts: np.ndarray, window: int):
	"""
	Computes the agreement matrix between two sortings.
	Looks at all pairs of clusters, and computes the number of coincident spikes.

	@params roots1:
		The root of the first sorting.
	@params roots2:
		The root of the second sorting.
	@params window (int):
		The number of time intervals to consider 2 spikes coincident.
	"""

	agreement_matrix = np.zeros([len(roots1), len(roots2), 2], dtype=np.float32)

	for i in range(len(roots1)):
		for j in range(len(roots2)):
			spike_train1 = roots1[i]
			spike_train2 = roots2[j] - shifts[i, j]

			n_coincidents = utils.get_nb_coincident_spikes(spike_train1, spike_train2, window)
			agreement_matrix[i, j, 0] = n_coincidents / len(spike_train1)
			agreement_matrix[i, j, 1] = n_coincidents / len(spike_train2)

	return agreement_matrix


def _correlogram_validation(data: PhyData, similar_units: np.ndarray, params: dict, plot_folder: str):
	"""
	Checks if the similar clusters come from a single unit based of the auto/cross-correlograms.

	@params data (PhyData):
		The data object.
	@params similar_units (np.ndarray) [n_units, 2]:
		Pairs of clusters that are considered similar.
	@params params (dict):
		The parameters for the correlogram validation step.
	@params plot_folder (str):
		Path to the logs folder for plots.
	"""

	indices_to_delete = []
	b, a = filter.get_filter_params(params['filter'][0], max_f=params['filter'][1], sampling_rate=1e3/params['bin_size'], btype="lowpass")

	correlograms = []
	pairs_to_plot = []
	differences = []
	windows = []

	for i in range(len(similar_units)):
		unit1 = similar_units[i, 0]
		unit2 = similar_units[i, 1]
		spike_train1 = data._sortings[unit1[0]].get_unit_spike_train(unit1[1])
		spike_train2 = data._sortings[unit2[0]].get_unit_spike_train(unit2[1])

		auto_corr1 = utils.get_autocorr_from_spiketrain(spike_train1, bin_size=params['bin_size'], max_time=params['max_time'])[0].astype(np.int32)
		auto_corr2 = utils.get_autocorr_from_spiketrain(spike_train2, bin_size=params['bin_size'], max_time=params['max_time'])[0].astype(np.int32)
		cross_corr = utils.get_crosscorr_from_spiketrain(spike_train1, spike_train2, bin_size=params['bin_size'], max_time=params['max_time'])[0].astype(np.int32)
		m = len(auto_corr1) // 2
		auto_corr1[m-1:m+1] = auto_corr2[m-1:m+1] = cross_corr[m-1:m+1] = 0

		N1 = len(spike_train1)
		N2 = len(spike_train2)
		W = merge_units.get_units_window(np.array([auto_corr1, auto_corr2]), params['bin_size'], params['window'])
		w = int(round((N1*W[0] + N2*W[1]) / (N1+N2)))

		plage = np.arange(m-w, m+w).astype(np.uint16)
		auto_corr1 = filter.filter(auto_corr1, b, a, np.float32)
		auto_corr2 = filter.filter(auto_corr2, b, a, np.float32)
		cross_corr = filter.filter(cross_corr, b, a, np.float32)

		diff1, diff2 = merge_units.get_correlogram_differences(auto_corr1, auto_corr2, cross_corr, plage)
		diff = (N1*diff1 + N2*diff2) / (N1+N2)

		if diff > 1 - params['similarity']:
			indices_to_delete.append(i)

		if diff < 1.1 - params['similarity']:
			correlograms.append([auto_corr1, auto_corr2, cross_corr])
			pairs_to_plot.append([unit1, unit2])
			differences.append(diff)
			windows.append(w)

	correlograms = np.array(correlograms, dtype=np.float32)
	pairs_to_plot = np.array(pairs_to_plot, dtype=np.uint16)
	differences = np.array(differences, dtype=np.float32)
	windows = np.array(windows, dtype=np.float32) * params['bin_size']
	_plot_correlogram_checks(correlograms, pairs_to_plot, differences, differences < 1-params['similarity'], params['max_time']-params['bin_size']/2, windows, plot_folder)

	return np.delete(similar_units, indices_to_delete, axis=0)


def _waveform_validation(data: PhyData, similar_units: np.ndarray, shifts: np.ndarray, max_shift: float, params: dict, plot_folder: str):
	"""
	Checks if the similar clusters come from a single unit based of their mean waveforms.

	@params data (PhyData):
		The data object.
	@params similar_units (np.ndarray) [n_units, 2]:
		Pairs of clusters that are considered similar.
	@params shifts (dict):
		shifts[i][j] contains a 2D array with all of the shifts between the clusters of sortings i and j.
	@param max_shift (float):
		The maximum allowed shift (in ms).
	@params params (dict):
		The parameters for the waveform validation step.
	@params plot_folder (str):
		Path to the logs folder for plots.
	"""

	indices_to_delete = []
	max_shift_samples = int(max_shift * data.sampling_f * 1e-3)
	b, a = filter.get_filter_params(params['filter'][0], params['filter'][1], params['filter'][2], btype="bandpass")

	to_plot_wvfs = []
	to_plot_channels = []
	to_plot_scores = []

	for i in range(len(similar_units)):
		unit1 = similar_units[i, 0]
		unit2 = similar_units[i, 1]
		shift = shifts[i]

		data.set_sorting(unit1[0])
		wvf_1 = data.get_unit_mean_waveform(unit1[1], ms_before=params['waveforms']['ms_before']+max_shift, ms_after=params['waveforms']['ms_after']+max_shift, max_spikes_per_unit=params['waveforms']['max_spikes_per_unit'])
		wvf_1 = filter.filter(wvf_1, b, a, np.float32)[:, max_shift_samples : -max_shift_samples]

		data.set_sorting(unit2[0])
		wvf_2 = data.get_unit_mean_waveform(unit2[1], ms_before=params['waveforms']['ms_before']+max_shift, ms_after=params['waveforms']['ms_after']+max_shift, max_spikes_per_unit=params['waveforms']['max_spikes_per_unit'])
		wvf_2 = filter.filter(wvf_2, b, a, np.float32)[:, max_shift_samples-shift : wvf_2.shape[1]-max_shift_samples-shift]

		score, channels = merge_units.compute_waveforms_difference(wvf_1, wvf_2, n_max_channels=params['waveforms']['max_channels_per_waveforms'], max_shift=0, return_channels=True)

		if score > 1 - params['similarity']:
			indices_to_delete.append(i)

		to_plot_wvfs.append(wvf_1)
		to_plot_wvfs.append(wvf_2)
		to_plot_channels.append(channels)
		to_plot_scores.append(score)

	to_plot_wvfs = np.array(to_plot_wvfs, dtype=np.float32)
	to_plot_channels = np.array(to_plot_channels, dtype=np.uint16)
	to_plot_scores = np.array(to_plot_scores, dtype=np.float32)
	_plot_waveform_checks(to_plot_wvfs, similar_units, shifts, to_plot_channels, to_plot_scores, to_plot_scores <= 1-params['similarity'], (params['waveforms']['ms_before'], params['waveforms']['ms_after']), data.uvolt_ratio, plot_folder)

	return np.delete(similar_units, indices_to_delete, axis=0)


def perform_final_merging(data: PhyData, similar_units: list, shifts: list, params: dict, plot_folder: str):
	"""
	Performs the merging between the similar units based on highest score.

	@params data (PhyData):
		The data object.
	@params similar_units (list of list) [n_units, n_clusters_per_unit]:
		List of potential units, with all of the clusters corresponding to this unit.
	@params shifts (dict):
		shifts[i][j] contains a 2D array with all of the shifts between the clusters of sortings i and j.
	@params params (dict):
		The parameters for merge_sortings.
	@params plot_folder (str):
		Path to the logs folder for plots.
	"""

	t_max = data.recording.get_num_frames()
	logs = open("{0}/final_merging.logs".format(plot_folder), 'x')

	for i in range(len(similar_units)):
		cluster = similar_units[i]
		score = -4000
		spikes = None
		units = []
		new_unit_id = data.merged_sorting.get_unit_ids()[-1] + 1 if len(data.merged_sorting.get_unit_ids()) > 0 else 0
		logs.write("\nMaking unit {0} from\n".format(new_unit_id))
		logs.write("Cluster: {0}\n".format(np.array(cluster).tolist()))

		for n_units in range(1, min(1+len(cluster), 2)):
			for indices in itertools.combinations(range(len(cluster)), n_units):
				combination = [cluster[x] for x in indices]
				tmp_shifts = np.array(shifts[i], dtype=np.int32)[list(indices)]
				if n_units > 1:
					spike_train = np.sort(np.unique(list(itertools.chain(*[data._sortings[combination[i][0]].get_unit_spike_train(combination[i][1]).astype(np.int64) - tmp_shifts[i] for i in range(len(combination))]))))
					spike_train = np.where(spike_train < 0, 0, spike_train).astype(np.uint64)
				else:
					spike_train = data._sortings[combination[0][0]].get_unit_spike_train(combination[0][1])
				duplicates = np.argwhere(np.diff(spike_train) <= params['duplicated_spike'])[:, 0]
				spike_train = np.delete(spike_train, duplicates+1)

				F = len(spike_train) / t_max * data.sampling_f
				C = utils.estimate_spike_train_contamination(spike_train, tuple(params['refractory_period']), t_max, data.sampling_f)
				S = F * (1 - 3.5*C)
				logs.write("\tCombination {0} -->\n\tScore = {1:.2f}\n\t-----------------\n".format(np.array(combination).tolist(), S))
				if np.sum(spike_train < 0) > 0:
					print("Error: spike train negative in merge_sortings.perform_final_merging()")
					exit()

				if S > score:
					score = S
					spikes = spike_train
					units = combination

		logs.write("Chose units {0}\n".format(units))

		data.merged_sorting.add_unit(new_unit_id, spikes.astype(np.uint64))

		for unit in cluster:
			data._sortings[unit[0]].exclude_units([unit[1]])

	logs.close()


def check_conflicts(data: PhyData, starting_ID: int, params: dict, plot_folder: str):
	"""
	Checks for conflics (coincident spikes) between clusters in the merged sorting.
	Deletes the duplicated or "wrong" clusters.

	@params data (PhyData):
		The data object.
	@params starting_ID (int):
		The starting unit_id of the merged sorting (to not check with units not of this run).
	@params params (dict):
		The parameters for merge_sortings.
	@params plot_folder (str):
		Path to the logs folder for plots.
	"""

	t_max = data.recording.get_num_frames()
	max_shift_samples = int(params['max_reshift'] * data.sampling_f * 1e-3)
	shared_clusters = []

	wvf_extraction_kwargs = {}
	wvf_extraction_kwargs['ms_before'] = params['waveform_validation']['waveforms']['ms_before'] + params['max_reshift']
	wvf_extraction_kwargs['ms_after']  = params['waveform_validation']['waveforms']['ms_after']  + params['max_reshift']
	wvf_extraction_kwargs['max_spikes_per_unit'] = 1000
	wvf_extraction_kwargs['max_channels_per_waveforms'] = None
	b, a = filter.get_filter_params(params['waveform_validation']['filter'][0], params['waveform_validation']['filter'][1], params['waveform_validation']['filter'][2], btype="bandpass")

	mean_wvfs = dict()
	for unit_id in data.merged_sorting.get_unit_ids():
		if unit_id < starting_ID:
			continue

		spike_train = data.merged_sorting.get_unit_spike_train(unit_id).astype(np.uint64)
		mean_wvf = np.mean(data.get_waveforms_from_spiketrain(spike_train, **wvf_extraction_kwargs), axis=0, dtype=np.float32)
		mean_wvf = filter.filter(mean_wvf, b, a, dtype=np.float32)
		mean_wvfs[unit_id] = mean_wvf

	for i in range(len(data.merged_sorting.get_unit_ids())):
		unit_id_1 = data.merged_sorting.get_unit_ids()[i]
		if unit_id_1 < starting_ID:
				continue

		spike_train1 = data.merged_sorting.get_unit_spike_train(unit_id_1).astype(np.uint64)
		mean_wvf1 = mean_wvfs[unit_id_1]

		for j in range(i+1, len(data.merged_sorting.get_unit_ids())):
			unit_id_2 = data.merged_sorting.get_unit_ids()[j]

			spike_train2 = data.merged_sorting.get_unit_spike_train(unit_id_2)
			mean_wvf2 = mean_wvfs[unit_id_2]
			
			#best_channels = np.argsort(np.max(np.abs(mean_wvf1)+np.abs(mean_wvf2), axis=1))[::-1]
			conv = np.sum(scipy.signal.fftconvolve(mean_wvf1[:, ::-1], mean_wvf2[:, max_shift_samples:-max_shift_samples], mode="valid", axes=1), axis=0)
			shift = np.argmax(conv) - max_shift_samples

			spike_train2 = spike_train2.astype(np.int64) + shift
			spike_train2 = np.where(spike_train2 < 0, 0, spike_train2).astype(np.uint64)

			n_coincidents = utils.get_nb_coincident_spikes(spike_train1, spike_train2, 1)
			shared = max(n_coincidents / len(spike_train1), n_coincidents / len(spike_train2))

			if shared > 0.15:
				shared_clusters.append([unit_id_1, unit_id_2])

	units_to_delete = []

	for shared_cluster in shared_clusters:
		if shared_cluster[0] in units_to_delete or shared_cluster[1] in units_to_delete:
			continue

		scores = np.zeros([2], dtype=np.float32)

		for i in range(2):
			spike_train = data.merged_sorting.get_unit_spike_train(shared_cluster[i])
			F = len(spike_train) / t_max * data.sampling_f
			C = utils.estimate_spike_train_contamination(spike_train, tuple(params['refractory_period']), t_max, data.sampling_f)
			scores[i] = F * (1 - 5*C)

		best_unit = shared_cluster[np.argmax(scores)]
		utils.plot_units_from_spiketrain(data, [data.merged_sorting.get_unit_spike_train(unit_id) for unit_id in shared_cluster], shared_cluster, plot_folder, filename="cluster-{0}_({1})".format('-'.join([str(a) for a in shared_cluster]), best_unit))

		for unit in shared_cluster:
			if unit == best_unit:
				continue

			units_to_delete.append(unit)
	
	for unit_to_delete in units_to_delete:
		del data.merged_sorting._units[unit_to_delete]


def add_leftover_units(data: PhyData, unit_ids: dict, params: dict, plot_folder: str):
	"""
	Add clusters that didn't find any "friend" in other sortings,
	if they don't conflict with any already existing clusters in the merged sorting.

	@params data (PhyData):
		The data object.
	@params unit_ids (dict):
		Dictionnary with key=index of sorting in sortings list ; and values = list of unit ids to compute.
	@params params (dict):
		The parameters for merge_sortings.
	@params plot_folder (str):
		Path to the logs folder for plots.
	"""

	logs = open("{0}/add_leftover_units.logs".format(plot_folder), 'x')

	max_shift_samples = int(params['max_reshift'] * data.sampling_f * 1e-3)
	wvf_extraction_kwargs = {}
	wvf_extraction_kwargs['ms_before'] = params['waveform_validation']['waveforms']['ms_before'] + params['max_reshift']
	wvf_extraction_kwargs['ms_after']  = params['waveform_validation']['waveforms']['ms_after']  + params['max_reshift']
	wvf_extraction_kwargs['max_spikes_per_unit'] = 1000
	wvf_extraction_kwargs['max_channels_per_waveforms'] = None
	b, a = filter.get_filter_params(params['waveform_validation']['filter'][0], params['waveform_validation']['filter'][1], params['waveform_validation']['filter'][2], btype="bandpass")

	mean_wvfs = dict()
	merged_unit_ids = data.merged_sorting.get_unit_ids()
	for unit_id in merged_unit_ids:
		spike_train = data.merged_sorting.get_unit_spike_train(unit_id).astype(np.uint64)
		mean_wvf = np.mean(data.get_waveforms_from_spiketrain(spike_train, **wvf_extraction_kwargs), axis=0, dtype=np.float32)
		mean_wvf = filter.filter(mean_wvf, b, a, dtype=np.float32)
		mean_wvfs[unit_id] = mean_wvf

	for sorting in unit_ids.keys():
		logs.write("\nSorting {0}:\n".format(sorting))
		data.set_sorting(sorting)
		units = data._sortings[sorting].get_unit_ids()
		unit_ids[sorting] = np.array([unit_id for unit_id in unit_ids[sorting] if unit_id in units], dtype=np.uint16)

		for unit_id in unit_ids[sorting]:
			logs.write("\t- Unit {0}: ".format(unit_id))
			conflict = False

			mean_wvf = data.get_unit_mean_waveform(unit_id, **wvf_extraction_kwargs)
			mean_wvf = filter.filter(mean_wvf, b, a, dtype=np.float32)

			for unit in merged_unit_ids:
				conv = np.sum(scipy.signal.fftconvolve(mean_wvfs[unit][:, ::-1], mean_wvf[:, max_shift_samples:-max_shift_samples], mode="valid", axes=1), axis=0)
				shift = np.argmax(conv) - max_shift_samples

				spike_train = data._sortings[sorting].get_unit_spike_train(unit_id) + shift
				spike_train = np.where(spike_train < 0, 0, spike_train).astype(np.uint64)

				spike_train2 = data.merged_sorting.get_unit_spike_train(unit).astype(np.uint64)
				n_coincidents = utils.get_nb_coincident_spikes(spike_train, spike_train2, params['similarity_validation']['window'])
				overlap = max(n_coincidents / len(spike_train), n_coincidents / len(spike_train2))
				if overlap > 0.2:
					conflict = True
					logs.write("Overlap with unit {0}: {1:.2f} %\n".format(unit, 100*overlap))
					break

			if conflict:
				continue # Skip unit.

			new_unit_id = data.merged_sorting.get_unit_ids()[-1] + 1 if len(data.merged_sorting.get_unit_ids()) > 0 else 0
			logs.write("No overlap, unit added (n°{0}).\n".format(new_unit_id))
			data.merged_sorting.add_unit(new_unit_id, data._sortings[sorting].get_unit_spike_train(unit_id))

		data._sortings[sorting].exclude_units(unit_ids[sorting])

	logs.close()


def _get_cross_cluster_shift(data: PhyData, unit_ids: dict, params: dict):
	"""
	Computes the shift between clusters between all cluster between all sortings.

	@params data (PhyData):
		The data object.
	@params unit_ids (dict):
		Dictionary containing all of the unit_ids (values) for all sortings (keys).
	@params params (dict):
		The parameters for merge_sortings.
	"""

	shifts = dict()

	max_shift = params['max_reshift']
	max_shift_samples = int(max_shift * data.sampling_f * 1e-3)

	b, a = filter.get_filter_params(params['waveform_validation']['filter'][0], params['waveform_validation']['filter'][1], params['waveform_validation']['filter'][2], btype="bandpass")
	mean_waveforms = dict()
	for sorting in unit_ids.keys():
		data.set_sorting(sorting)
		mean_waveforms[sorting] = data.get_units_mean_waveform(unit_ids[sorting], ms_before=params['waveform_validation']['waveforms']['ms_before'] + max_shift,
									ms_after=params['waveform_validation']['waveforms']['ms_after'] + max_shift, max_spikes_per_unit=params['waveform_validation']['waveforms']['max_spikes_per_unit'],
									max_channels_per_waveforms=None)
		mean_waveforms[sorting] = filter.filter(mean_waveforms[sorting], b, a, dtype=np.float32)

	for sorting1 in unit_ids.keys():
		shifts[sorting1] = dict()

		for sorting2 in unit_ids.keys():
			shifts[sorting1][sorting2] = np.zeros([len(unit_ids[sorting1]), len(unit_ids[sorting2])], dtype=np.int16)

			for i in range(len(unit_ids[sorting1])):
				conv = center_waveform.compute_convolution(mean_waveforms[sorting2], mean_waveforms[sorting1][i, :, max_shift_samples:-max_shift_samples])
				shifts[sorting1][sorting2][i] = -(np.argmax(conv, axis=1) - max_shift_samples)

	return shifts




def _compute_agreement_matrices_parallel(sortings: list, unit_ids: dict, shifts: dict, window: int, n_process: int=8):
	"""
	Parallel implementation of merge_sortings._compute_agreement_matrices().
	"""
	if n_process == 1:
		return _compute_agreement_matrices(sortings, unit_ids, window)

	agreement_matrices = dict()

	with Pool(processes=n_process) as pool:
		res = []

		for i in list(unit_ids.keys())[:-1]:
			for j in unit_ids.keys():
				if j <= i:
					continue

				roots1 = [root.get_spike_train().astype(np.uint64) for root in sortings[i]._roots if root.unit_id in unit_ids[i]]
				roots2 = [root.get_spike_train().astype(np.uint64) for root in sortings[j]._roots if root.unit_id in unit_ids[j]]
				res.append(pool.apply_async(_compute_agreement_matrix, (roots1, roots2, shifts[i][j], window)))

		idx = 0
		for i in list(unit_ids.keys())[:-1]:
			agreement_matrices[i] = dict()

			for j in unit_ids.keys():
				if j <= i:
					continue

				agreement_matrices[i][j] = res[idx].get()
				idx += 1

	return agreement_matrices




def _plot_all_units(data: PhyData, unit_ids: dict, params: dict, plot_folder: str):
	"""

	"""

	for sorting, units in unit_ids.items():
		data.set_sorting(sorting)
		utils.plot_units(data, units, plot_folder, filename="sorting_{0}_units".format(sorting), **params)


def _plot_correlogram_checks(correlograms: np.ndarray, pairs: np.ndarray, differences: np.ndarray, passed: list, max_time: float, windows: np.ndarray, plot_folder: str):
	"""
	Plots the result of the _correlogram_validation function.

	@param correlograms (np.ndarray) [n_pairs, 3, time]:
		Auto- and cross-correlograms of all the pairs of units (auto_corr1, auto_corr2, cross_corr).
	@param pairs (np.ndarray[int]) [n_pairs, 2, 2]:
		The units by pair (sorting index + unit id).
	@param differences (np.ndarray[float]):
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

	if len(pairs) == 0 or len(correlograms) == 0:
		return

	os.makedirs(plot_folder, exist_ok=True)

	fig = make_subplots(rows=1, cols=2, shared_xaxes=True,
						subplot_titles=("Auto-correlograms", "Cross-correlogram"))
	steps = []
	xaxis = np.linspace(-max_time, max_time, correlograms.shape[2])

	fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
	fig.update_xaxes(title_text="Time (ms)", row=1, col=2)

	for i in range(len(pairs)):
		fig.add_trace(go.Scatter(
			x=xaxis,
			y=correlograms[i, 0],
			mode="lines",
			name="Unit {1} (from sorting {0})".format(*pairs[i, 0]),
			marker_color="CornflowerBlue",
			visible=False
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis,
			y=correlograms[i, 1],
			mode="lines",
			name="Unit {1} (from sorting {0})".format(*pairs[i, 1]),
			marker_color="LightSeaGreen",
			visible=False
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis,
			y=correlograms[i, 2],
			mode="lines",
			name="Cross-correlogram",
			marker_color="Crimson",
			visible=False
		), row=1, col=2)

		fig.add_trace(go.Scatter(x=[-windows[i]]*2, y=[0, max(np.max(correlograms[i, 0]), np.max(correlograms[i, 1]))], mode="lines", marker_color="red", name="Check window", visible=False), row=1, col=1)
		fig.add_trace(go.Scatter(x=[+windows[i]]*2, y=[0, max(np.max(correlograms[i, 0]), np.max(correlograms[i, 1]))], mode="lines", marker_color="red", name="Check window", showlegend=False, visible=False), row=1, col=1)
		fig.add_trace(go.Scatter(x=[-windows[i]]*2, y=[0, np.max(correlograms[i, 2])], mode="lines", marker_color="red", name="Check window", showlegend=False, visible=False), row=1, col=2)
		fig.add_trace(go.Scatter(x=[+windows[i]]*2, y=[0, np.max(correlograms[i, 2])], mode="lines", marker_color="red", name="Check window", showlegend=False, visible=False), row=1, col=2)

		step = dict(
			label="{0}-{1}&{2}-{3}".format(pairs[i, 0, 0], pairs[i, 0, 1], pairs[i, 1, 0], pairs[i, 1, 1]),
			method="update",
			args=[
				{"visible": [j//7 == i for j in range(7*len(pairs))]},
				{"title.text": "Units {0}-{1} & {2}-{3} (Difference = {4:.2f})".format(pairs[i, 0, 0], pairs[i, 0, 1], pairs[i, 1, 0], pairs[i, 1, 1], differences[i]),
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
	fig.write_html("{0}/correlograms_validation.html".format(plot_folder))


def _plot_waveform_checks(waveforms: np.ndarray, pairs: np.ndarray, shifts: np.ndarray, channels: np.ndarray, scores: np.ndarray, passed: list, time_bound: tuple, uvolt_ratio, plot_folder: str):
	"""
	Plots the result of the _waveform_validation function.

	@param waveforms (np.ndarray) [n_units=2*n_pairs, n_tot_channels, time]:
		Filtered mean waveform of all units.
	@param pairs (np.ndarray[int]) [n_pairs, 2, 2]:
		The units by pair (sorting index + unit id).
	@param channels (np.ndarray[int]) [n_pairs, n_channels]
		Channels used for comparison for each pair.
	@param scores (np.ndarray[float]) [n_pairs]:
		The computed difference between the two mean waveforms.
	@param passed (list[bool]) [n_pairs]:
		Did the waveform check pass.
	@param time_bound (tuple of 2 float):
		Time axis bound for plot (in ms).
	@param uvolt_ratio (float or None):
		Ratio to go to µV. None for arbitrary units.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	if len(waveforms) == 0 or len(pairs) == 0:
		return

	os.makedirs(plot_folder, exist_ok=True)

	n_channels = channels.shape[1]
	n_rows = n_channels // 3
	n_cols = math.ceil(n_channels / n_rows)
	fig = make_subplots(rows=n_rows, cols=n_cols, shared_yaxes=True)
	for row in range(1, n_rows+1):
		for col in range(1, n_cols+1):
			fig.update_xaxes(title_text="Time (ms)", row=row, col=col)
			fig.update_yaxes(title_text="Voltage ({0})".format("A.U." if uvolt_ratio is None else "µV"), row=row, col=col)

	steps = []
	xaxis = np.linspace(-time_bound[0], time_bound[1], waveforms.shape[2])

	for i in range(len(pairs)):
		for channel in range(n_channels):
			row = 1 + channel//n_cols
			col = 1 + channel%n_cols

			fig.add_trace(go.Scatter(
				x=xaxis,
				y=waveforms[2*i, channels[i, channel]] * (1 if uvolt_ratio is None else uvolt_ratio),
				mode="lines",
				marker_color="CornflowerBlue",
				name="Unit {0}-{1} (channel {2})".format(*pairs[i, 0], channels[i, channel]),
				visible=False
			), row=row, col=col)
			fig.add_trace(go.Scatter(
				x=xaxis,
				y=waveforms[2*i+1, channels[i, channel]] * (1 if uvolt_ratio is None else uvolt_ratio),
				mode="lines",
				marker_color="LightSeaGreen",
				name="Unit {0}-{1} (channel {2})".format(*pairs[i, 1], channels[i, channel]),
				visible=False
			), row=row, col=col)

		step = dict(
			label="{0}-{1}&{2}-{3}".format(*pairs[i, 0], *pairs[i, 1]),
			method="update",
			args=[
				{"visible": [j//(2*n_channels) == i for j in range(2*len(pairs)*n_channels)]},
				{"title.text": "Units {0}-{1} & {2}-{3} (Difference = {4:.2f})".format(*pairs[i, 0], *pairs[i, 1], scores[i]),
				"title.font.color": "black" if passed[i] else "red",
				"annotations": [
					dict(x=0.9, y=1.1, xref="paper", yref="paper", showarrow=False,
						text="Shift = {0} pt".format(shifts[i]))
				]}
			]
		)
		steps.append(step)

	for i in range(2*n_channels):
		fig.data[i].visible = True
	sliders = [dict(
		active=0,
		currentvalue={"prefix": "Units "},
		pad={"t": 50},
		steps=steps
	)]

	fig.update_layout(width=1440, height=810, sliders=sliders)
	fig.write_html("{0}/waveform_validation.html".format(plot_folder))
