import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pytest
import lussac.main
from conftest import params_path


if not os.path.exists(params_path):
	http_response = urlopen("https://zenodo.org/record/7007074/files/lussac2_cerebellar_cortex_dev.zip")
	zip_file = ZipFile(BytesIO(http_response.read()))
	zip_file.extractall(path="tests/datasets")


def test_parse_arguments() -> None:
	with pytest.raises(SystemExit):  # There is no arguments.
		lussac.main.parse_arguments(None)

	assert lussac.main.parse_arguments([params_path]) == params_path
