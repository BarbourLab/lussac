import pytest
from lussac.core.lussac_data import LussacData, MonoSortingData
from lussac.core.pipeline import LussacPipeline
import lussac.main


params_path = "tests/datasets/cerebellar_cortex/params.json"


@pytest.fixture(scope="session")
def params() -> dict:
	return lussac.main.load_json(params_path)


@pytest.fixture(scope="session")
def data(params: dict) -> LussacData:
	return LussacData.create_from_params(params)


@pytest.fixture(scope="session")
def mono_sorting_data(data: LussacData) -> MonoSortingData:
	return MonoSortingData(data, 'ms3_best')


@pytest.fixture(scope="session")
def pipeline(data: LussacData) -> LussacPipeline:
	return LussacPipeline(data)
