import numpy as np

from phy_data import PhyData
import postprocessing.filter as filter


def remove_duplicated_spikes(data: PhyData, unit_ids: list, params: dict, plot_folder: str):
	"""
	Removes the duplicated spikes from the same unit.

	@param data (PhyData):
		The data object.
	@param unit_ids (list):
		List of units' id to check for duplicated spikes.
	@param params (dict):
		Parameters for the remove_duplicates function.
	@param plot_folder (str):
		Path to the plot folder.
	"""

	kwargs = {
		"unit_ids": unit_ids,
		"window": 0.2,
		"filter": [2, 300, 6000],
		"waveforms": {
			"ms_before": 1.0,
			"ms_after": 1.0,
			"max_channels_per_waveforms": 5
		}
	}
	kwargs.update(params)

	for i in range(len(unit_ids)):
		remove_unit_duplicated_spikes(data, unit_ids[i], kwargs)


def remove_unit_duplicated_spikes(data: PhyData, unit_id: int, params: dict):
	"""
	Removes the duplicated spikes from the unit.

	@param data (PhyData):
		Tha data object.
	@param unit_id (int):
		ID of unit to look for duplicated spikes.
	@param params (dict):
		Parameters for the remove_duplicates function.
	"""

	spike_train = data.get_unit_spike_train(unit_id)
	spike_train = np.unique(spike_train)

	diff = np.diff(spike_train)
	w = params['window'] * 1e-3 * data.sampling_f
	duplicates = np.argwhere(diff <= w)[:, 0]

	if len(duplicates) > 0:
		spikes_to_delete = _discriminate_duplicates(data, unit_id, duplicates, params)
		spike_train = np.delete(spike_train, spikes_to_delete)

	data.set_unit_spike_train(unit_id, spike_train)


def _discriminate_duplicates(data: PhyData, unit_id: int, duplicates_train: np.ndarray, params: dict):
	"""
	Discriminates between the first and second spike to know which one to delete.

	@param data (PhyData):
		The data object.
	@param unit_id (int):
		ID of unit.
	@param duplicates_train (np.ndarray):
		Spike timings of the first of each pair or duplicates.
	@param params (dict):
		Parameters for the remove_duplicates function.

	@return spikes_to_delete (np.ndarray):
		Indices of spikes to delete from duplicates_train.
	"""

	spikes_to_delete = np.zeros([len(duplicates_train)], dtype=np.uint64)

	waveforms, best_channels = data.get_unit_waveforms(unit_id, **params['waveforms'], max_spikes_per_unit=5000, return_idx=True)
	mean_wvf = np.mean(waveforms, axis=0)

	spike_train = np.zeros([2*len(duplicates_train)], dtype=np.uint64)
	spike_train[::2] = duplicates_train
	spike_train[1::2] = duplicates_train+1

	b, a = filter.get_filter_params(params['filter'][0], params['filter'][1], params['filter'][2], btype="bandpass")
	waveforms = data.get_waveforms_from_spiketrain(spike_train, ms_before=params['waveforms']['ms_before'], ms_after=params['waveforms']['ms_after'], max_channels_per_waveforms=best_channels)
	waveforms = filter.filter(waveforms, b, a, dtype=np.float32)
	mean_wvf = filter.filter(mean_wvf, b, a, dtype=np.float32)

	for i in range(len(spikes_to_delete)):
		wf_1 = np.sum(waveforms[2*i] * mean_wvf)
		wf_2 = np.sum(waveforms[2*i+1] * mean_wvf)

		spikes_to_delete[i] = duplicates_train[i] + (0 if wf_1 >= wf_2 else 1)

	return spikes_to_delete

