import os
import pytest
from lussac.core.lussac_data import LussacData
from tests.test_main import params


@pytest.fixture(scope="session")
def data(params: dict) -> LussacData:
	return LussacData.create_from_params(params)


def test_create_from_params(data: LussacData) -> None:
	assert isinstance(data, LussacData)


def test_tmp_folder(data: LussacData) -> None:
	assert os.path.exists(data.tmp_folder)
	assert os.path.isdir(data.tmp_folder)


def test_sampling_f(data: LussacData) -> None:
	assert data.sampling_f == 30000


def test_num_sortings(data: LussacData) -> None:
	assert data.num_sortings == 7