import os
import pathlib
import shutil
import sys
import pytest
from lussac.core import LussacData, LussacPipeline, MonoSortingData, MultiSortingsData
import lussac.main


sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))  # Otherwise the tests are not found properly with 'pytest'.
params_path = pathlib.Path(__file__).parent / "datasets" / "cerebellar_cortex" / "params.json"


@pytest.fixture(scope="session")
def params() -> dict:
	return lussac.main.load_json(str(params_path.resolve()))


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


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
	# test "test_dataset_exists" has to be run first!
	for item in items.copy():
		if item.parent.name == "test_main.py":
			items.remove(item)
			items.insert(0, item)


def pytest_sessionstart(session: pytest.Session) -> None:
	# Remove lussac folder if exists.
	shutil.rmtree("tests/datasets/cerebellar_cortex/lussac", ignore_errors=True)
	os.makedirs("tests/tmp", exist_ok=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
	# Remove temporary directories
	shutil.rmtree("tests/datasets/cerebellar_cortex/tmp_*", ignore_errors=True)
	shutil.rmtree("tests/tmp", ignore_errors=True)
