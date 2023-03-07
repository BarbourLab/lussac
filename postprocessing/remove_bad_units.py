import os
import numpy as np

from phy_data import PhyData
import postprocessing.filter as filter
import postprocessing.utils as utils


def rm_bad_units(data: PhyData, unit_ids: list, params: dict, plot_folder: str):
	"""
	Removes units from sorting based on some criteria.

	@param data:
		The data object.
	@param unit_ids (list):
		List of units' id to potentialy merge together.
	@param params (dict):
		Parameters for the remove_bad_units function.
	@param plot_folder (str):
		Path to the plot folder.

	@return bad_units (dict):
		TODO
	"""

	bad_units = dict()

	for unit_id in unit_ids:
		if 'all' in params:
			bad_units[unit_id] = -1
			continue

		if 'frequency' in params:
			frequency = data.get_unit_firing_rate(unit_id)

			if 'min' in params['frequency'] and frequency < params['frequency']['min']:
				bad_units[unit_id] = -1
				continue
			if 'max' in params['frequency'] and frequency > params['frequency']['max']:
				bad_units[unit_id] = -1
				continue

		if 'contamination' in params:
			contamination = utils.estimate_unit_contamination(data, unit_id, tuple(params['contamination']['refractory_period']))

			if contamination > params['contamination']['max']:
				bad_units[unit_id] = -1
				continue

		if 'amplitude' in params:
			assert data.uvolt_ratio is not None, "rm_bad_units() with 'amplitude' parameter only works if µV_ratio has been given."

			waveforms = data.get_unit_waveforms(unit_id, **params['amplitude']['waveforms'])
			mean_wvf = np.mean(waveforms, axis=0, dtype=np.float32)

			if 'filter' in params['amplitude']:
				b, a = filter.get_filter_params(*params['amplitude']['filter'], data.sampling_f)
				mean_wvf = filter.filter(mean_wvf, b, a, dtype=np.float32)

			mean_wvf = mean_wvf * data.uvolt_ratio
			amplitude = np.max(np.abs(mean_wvf))

			if 'min' in params['amplitude'] and amplitude < params['amplitude']['min']:
				bad_units[unit_id] = -1
				continue
			if 'max' in params['amplitude'] and amplitude > params['amplitude']['max']:
				bad_units[unit_id] = -1
				continue

		if 'std_amplitudes' in params:
			assert data.uvolt_ratio is not None, "rm_bad_units() with 'std_amplitudes' parameter only works if µV_ratio has been given."

			waveforms, channel_idx = data.get_unit_waveforms(unit_id, return_idx=True, **params['std_amplitudes']['waveforms'])
			mean_wvf = np.mean(waveforms, axis=0, dtype=np.float32)

			spike_train = data.get_unit_spike_train(unit_id)
			amplitudes = data.recording.get_traces()[channel_idx[0], spike_train] * data.uvolt_ratio
			std_amplitudes = np.std(amplitudes)

			if 'min' in params['std_amplitudes'] and std_amplitudes < params['std_amplitudes']['min']:
				bad_units[unit_id] = -1
				continue
			if 'max' in params['std_amplitudes'] and std_amplitudes > params['std_amplitudes']['max']:
				bad_units[unit_id] = -1
				continue

	unit_ids = list(bad_units.keys())
	_plot_bad_units(data, unit_ids, plot_folder)
	data.sorting.exclude_units(unit_ids)

	return bad_units




def _plot_bad_units(data: PhyData, unit_ids: list, plot_folder: str):
	"""
	Plots the bad units that are going to be removed.

	@param data (PhyData):
		The data object.
	@param unit_ids (list of int):
		IDs of units that are going to be deleted.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	if len(unit_ids) == 0:
		return

	save_folder = "{0}/results".format(plot_folder)
	#os.makedirs(save_folder, exist_ok=True)

	utils.plot_units(data, unit_ids, save_folder)
