import pathlib

from lussac.core import LussacParams


def test_load_default_params() -> None:
	params = LussacParams.load_default_params('params_synthetic', '/aze/')
	print(params)
	assert params['lussac']['tmp_folder'] == str(pathlib.Path("/aze") / "lussac" / "tmp")
