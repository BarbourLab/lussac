import importlib.util
import numpy as np
import pytest
from lussac.core import LussacData, LussacSpikeSorter
import lussac.utils as utils
import spikeinterface.core as si
from spikeinterface.core.testing import check_sortings_equal


@pytest.mark.skipif(importlib.util.find_spec("hdbscan") is None, reason="Testing spike sorting requires Python package 'hdbscan'.")
def test_spike_sorting(data: LussacData) -> None:
	"""
	Tests if spike sorting can be run through Lussac without any crash.
	Creates a shorter recording (1 shank, 5 minutes) to go faster.
	We don't really care about quality here (e.g. no preprocessing)
	because we only want to check that the spike sorting can be run.
	"""
	start_frame, end_frame = 9_000_000, 18_000_000
	name = "test_spike_sorting"

	recording = data.recording.frame_slice(start_frame, end_frame)
	recording = recording.channel_slice(np.arange(13, 26))
	params = {
		'sorter_name': "spykingcircus2",
		'preprocessing': {
			'filter': {'band': [150., 6000.], 'filter_order': 2, 'ftype': "bessel"},
			'common_reference': {'operator': "median"}
		},
		'sorter_params': {
			'detection': {'peak_sign': "neg", "detect_threshold": 7},
			'apply_preprocessing': False,
			'output_folder': f"{data.tmp_folder}/sc2_ss_test"
		}
	}

	spike_sorter = LussacSpikeSorter(recording, name)
	sorting = spike_sorter.launch(params)

	assert isinstance(sorting, si.BaseSorting)
	assert sorting.get_annotation("name") == name
	assert sorting.get_num_units() > 0

	# Creating a "ground truth" for the good simple spike (checking if it was found).
	gt_spike_train = data.sortings['ks2_best'].get_unit_spike_train(41).astype(np.int64)
	start = np.searchsorted(gt_spike_train, start_frame)
	end = np.searchsorted(gt_spike_train, end_frame)
	gt_spike_train = gt_spike_train[start:end] - start_frame

	found_ss = False
	for unit_id in sorting.unit_ids:
		spike_train = sorting.get_unit_spike_train(unit_id).astype(np.int64)
		n_coincident = utils.compute_nb_coincidence(spike_train, gt_spike_train, max_time=6)
		accuracy = n_coincident / (len(gt_spike_train) + len(spike_train) - n_coincident)

		if accuracy > 0.8:
			found_ss = True
			break

	# Testing loading a previously-run spike sorting.
	sorting2 = spike_sorter.launch(params)
	check_sortings_equal(sorting, sorting2, check_annotations=True, check_properties=True)

	assert found_ss
