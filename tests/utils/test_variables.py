from lussac.utils import Utils


def test_variables() -> None:
	assert Utils.sampling_frequency == 30000
	assert Utils.t_max == 30000 * 60 * 15
