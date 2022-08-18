import pytest
import lussac.main


params_path = "tests/datasets/cerebellar_cortex/params.json"


def test_parse_arguments():
	with pytest.raises(SystemExit): # There is no arguments.
		lussac.main.parse_arguments(None)

	assert lussac.main.parse_arguments([params_path]) == params_path


@pytest.fixture(scope="session")
def params() -> dict:
	return lussac.main.load_json(params_path)
