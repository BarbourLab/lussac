import shutil
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
	return MonoSortingData(data, data.sortings['ms3_best'])


@pytest.fixture(scope="session")
def pipeline(data: LussacData) -> LussacPipeline:
	return LussacPipeline(data)


def pytest_sessionstart(session: pytest.Session) -> None:
	# Remove lussac folder if exists.
	shutil.rmtree("tests/datasets/cerebellar_cortex/lussac", ignore_errors=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
	# Remove temporary directory
	shutil.rmtree("tests/datasets/cerebellar_cortex/tmp_*", ignore_errors=True)
