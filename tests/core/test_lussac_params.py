import pathlib

from lussac.core import LussacParams


def test_load_default_params() -> None:
	params = LussacParams.load_default_params('synthetic', '/aze/')
	assert str(pathlib.Path(params['lussac']['tmp_folder']).absolute()) == str(pathlib.Path("/aze").absolute() / "lussac" / "tmp")
