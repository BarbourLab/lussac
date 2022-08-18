from lussac.core.lussac_data import LussacData
from tests.test_main import params


def test_create_from_params(params: dict):
	data = LussacData.create_from_params(params)

	assert isinstance(data, LussacData)
