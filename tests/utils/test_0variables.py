# File requires '0' in its name to execute before the others, to make sure that 'Utils' variables are set.

from lussac.core import LussacData
from lussac.utils import Utils


def test_variables(data: LussacData) -> None:
	assert data is not None  # Call fixture data to load Utils if not already done.
	assert Utils.sampling_frequency == 30000
	assert Utils.t_max == 30000 * 60 * 15
