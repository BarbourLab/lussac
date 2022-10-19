import os
import spikeinterface.core as si
from lussac.core.lussac_data import LussacData, MultiSortingsData


def test_recording(multi_sortings_data: MultiSortingsData) -> None:
	assert isinstance(multi_sortings_data.recording, si.BaseRecording)


def test_num_sortings(data: LussacData, multi_sortings_data: MultiSortingsData) -> None:
	assert multi_sortings_data.num_sortings == len(data.sortings)


def test_tmp_folder(multi_sortings_data: MultiSortingsData) -> None:
	assert os.path.exists(multi_sortings_data.tmp_folder)
	assert os.path.isdir(multi_sortings_data.tmp_folder)




def test_logs_folder(multi_sortings_data: MultiSortingsData) -> None:
	assert os.path.exists(multi_sortings_data.logs_folder)
	assert os.path.isdir(multi_sortings_data.logs_folder)
