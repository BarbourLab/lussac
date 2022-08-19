import lussac.version


def test_version() -> None:
	assert isinstance(lussac.version.version, str)
