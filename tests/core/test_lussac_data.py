import os
import pytest
from lussac.core.lussac_data import LussacData


def test_format_params() -> None:
	with pytest.raises(Exception):
		LussacData._format_params({0: {}})
	with pytest.raises(Exception):
		LussacData._format_params({'aze': 1})
	with pytest.raises(Exception):
		LussacData._format_params({'aze': {2: {}}})
	with pytest.raises(Exception):
		LussacData._format_params({'aze': {'aze': 3}})

	params = {
		'module': {
			"cat1": {'a': 1},
			"cat2;cat3": {'b': 2}
		}
	}
	formatted_params = LussacData._format_params(params)
	assert 'cat2' in formatted_params['module']
	assert 'cat3' in formatted_params['module']
	assert 'cat2;cat3' not in formatted_params['module']
	formatted_params['module']['cat2']['b'] = 3
	assert formatted_params['module']['cat1']['a'] == 1
	assert formatted_params['module']['cat2']['b'] == 3
	assert formatted_params['module']['cat3']['b'] == 2


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


def test_create_from_params(data: LussacData) -> None:
	assert isinstance(data, LussacData)
