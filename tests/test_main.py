import os
import requests
import pathlib
import pytest
import shutil
from tqdm import tqdm
import lussac.main
from conftest import params_path
import spikeinterface.core as si


def test_dataset_exists(capsys):
	if not params_path.exists():
		with capsys.disabled():  # Don't capture the print output and show it.
			file_path = pathlib.Path(__file__).parent / "datasets" / "cerebellar_cortex.zip"
			file_path.parent.mkdir(parents=True, exist_ok=True)

			http_response = requests.get("https://zenodo.org/record/8171929/files/lussac2_cerebellar_cortex_dev.zip", stream=True)
			n_bytes = int(http_response.headers.get("content-length"))

			with tqdm.wrapattr(open(file_path, 'wb'), "write", miniters=1, desc=f"Downloading {file_path.name}", total=n_bytes) as fout:
				for chunk in http_response:
					fout.write(chunk)
				fout.close()  # Necessary even inside a 'with'.

			print("Unzipping ...")
			shutil.unpack_archive(file_path, file_path.parent)
			file_path.unlink()

			zarr_folder = file_path.parent / "cerebellar_cortex" / "recording.zarr"
			recording = si.ZarrRecordingExtractor(zarr_folder)
			recording.save(format="binary", folder=zarr_folder.parent / "recording.bin", n_jobs=2, chunk_duration='2s')

			print("")
			print(pathlib.Path(__file__).relative_to(pathlib.Path(os.getcwd())), end=' ')  # To have a nice output in the console.

	assert params_path.exists()


def test_parse_arguments() -> None:
	with pytest.raises(SystemExit):  # There is no arguments.
		lussac.main.parse_arguments(None)

	assert lussac.main.parse_arguments([str(params_path)]) == str(params_path)
