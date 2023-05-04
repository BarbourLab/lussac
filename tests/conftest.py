import os
import pathlib
import platform
import shutil
import sys
import pytest
from lussac.core import LussacData, LussacPipeline, MonoSortingData, MultiSortingsData
import lussac.main


sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))  # Otherwise the tests are not found properly with 'pytest'.
params_path = pathlib.Path(__file__).parent / "datasets" / "cerebellar_cortex" / "params.json"


def convert_params_to_windows(params: dict) -> dict:
	for key, value in params:
		if isinstance(value, dict):
			params[key] = convert_params_to_windows(value)
		elif isinstance(value, str):
			params[key] = params[key].replace('/', '\\')

	return params


@pytest.fixture(scope="session")
def params() -> dict:
	params = lussac.main.load_json(str(params_path.resolve()))
	if platform.system() == "Windows":
		convert_params_to_windows(params)

	return params


@pytest.fixture(scope="session")
def data(params: dict) -> LussacData:
	return LussacData.create_from_params(params)


@pytest.fixture(scope="session")
def mono_sorting_data(data: LussacData) -> MonoSortingData:
	return MonoSortingData(data, data.sortings['ms3_best'])


@pytest.fixture(scope="session")
def multi_sortings_data(data: LussacData) -> MultiSortingsData:
	return MultiSortingsData(data, data.sortings)


@pytest.fixture(scope="session")
def pipeline(data: LussacData) -> LussacPipeline:
	return LussacPipeline(data)


def pytest_sessionstart(session: pytest.Session) -> None:
	# Remove lussac folder if exists.
	shutil.rmtree("tests/datasets/cerebellar_cortex/lussac", ignore_errors=True)
	os.makedirs("tests/tmp", exist_ok=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
	# Remove temporary directories
	shutil.rmtree("tests/datasets/cerebellar_cortex/tmp_*", ignore_errors=True)
	shutil.rmtree("tests/tmp", ignore_errors=True)
