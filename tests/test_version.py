import lussac.version


def test_version():
	assert isinstance(lussac.version.version, str)
