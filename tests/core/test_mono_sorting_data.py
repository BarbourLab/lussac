import os
import spikeinterface.core as si
from lussac.core.lussac_data import LussacData, MonoSortingData


def test_recording(mono_sorting_data: MonoSortingData) -> None:
	assert isinstance(mono_sorting_data.recording, si.BaseRecording)


def test_sorting(data: LussacData, mono_sorting_data: MonoSortingData) -> None:
	assert isinstance(mono_sorting_data.sorting, si.BaseSorting)
	assert mono_sorting_data.sorting == data.sortings['ms3_best']


def test_name(mono_sorting_data: MonoSortingData) -> None:
	assert mono_sorting_data.name == "ms3_best"


def test_sampling_f(mono_sorting_data: MonoSortingData) -> None:
	assert mono_sorting_data.sampling_f == 30000


def test_tmp_folder(mono_sorting_data: MonoSortingData) -> None:
	assert os.path.exists(mono_sorting_data.tmp_folder)
	assert os.path.isdir(mono_sorting_data.tmp_folder)


def test_get_unit_spike_train(mono_sorting_data: MonoSortingData) -> None:
	assert len(mono_sorting_data.get_unit_spike_train(2)) == 1026
	assert len(mono_sorting_data.get_unit_spike_train(71)) == 62474
