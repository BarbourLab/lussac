import os
import numpy as np
from lussac.core import LussacData, MonoSortingData
from lussac.modules import AlignUnits
import spikeinterface.core as si


params = {
	'wvf_extraction': {
		'ms_before': 1.5,
		'ms_after': 2.5,
		'max_spikes_per_unit': 20,
		'filter_band': [150, 6000],
	},
	'threshold': 0.5,
	'check_next': 10
}


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = AlignUnits("test_align_units_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_align_units(mono_sorting_data: MonoSortingData) -> None:
	# Create a smaller data object for testing (faster).
	data = MonoSortingData(mono_sorting_data.data, mono_sorting_data.sorting.select_units([14, 22, 70, 71]))

	module = AlignUnits("test_align_units", data, "all")
	assert not os.path.exists(f"{module.logs_folder}/alignment.html")
	sorting = module.run(params)
	assert os.path.exists(f"{module.logs_folder}/alignment.html")

	assert sorting.get_num_units() == data.sorting.get_num_units()


def test_shift_sorting(data: LussacData) -> None:
	sorting = si.NumpySorting.from_unit_dict({0: np.array([5, data.recording.get_num_frames() - 5])}, sampling_frequency=data.recording.sampling_frequency)

	new_sorting = AlignUnits.shift_sorting(data.recording, sorting, {0: 2})
	assert new_sorting.get_num_units() == 1
	assert np.all(new_sorting.get_unit_spike_train(0) == (3, data.recording.get_num_frames() - 7))

	# Test that spikes over the edges are deleted.
	assert AlignUnits.shift_sorting(data.recording, sorting, {0: 10}).get_num_units() == 1
	assert AlignUnits.shift_sorting(data.recording, sorting, {0: -10}).get_num_units() == 1


def test_get_units_shift() -> None:
	templates = np.array([
		[0, 0, 0, 0, 0, 0, -20, 0],
		[50, 20, 10, 0, -10, -15, 0, 0],
		[0, 0, -20, -40, -30, -50, -10, 0],
		[0, -20, -40, 0, -10, -20, -50, -40]
	])

	shifts = AlignUnits.get_units_shift(templates, nbefore=3, threshold=0.5, check_next=0)
	assert np.all(shifts == (3, -3, 0, -1))

	shifts = AlignUnits.get_units_shift(templates, nbefore=3, threshold=0.5, check_next=2)
	assert np.all(shifts == (3, -3, 2, -1))

	shifts = AlignUnits.get_units_shift(templates, nbefore=3, threshold=0.5, check_next=10)
	assert np.all(shifts == (3, -3, 2, 3))
