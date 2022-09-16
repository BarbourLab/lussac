import os
from lussac.core.lussac_data import LussacData


def test_create_from_params(data: LussacData) -> None:
	assert isinstance(data, LussacData)


def test_tmp_folder(data: LussacData) -> None:
	assert os.path.exists(data.tmp_folder)
	assert os.path.isdir(data.tmp_folder)


def test_logs_folder(data: LussacData) -> None:
	assert os.path.exists(data.logs_folder)
	assert os.path.isdir(data.logs_folder)


def test_sampling_f(data: LussacData) -> None:
	assert data.sampling_f == 30000


def test_num_sortings(data: LussacData) -> None:
	assert data.num_sortings == 7
