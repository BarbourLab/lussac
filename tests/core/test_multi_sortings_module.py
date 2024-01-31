from typing import Any
from lussac.core import LussacData, MultiSortingsData, MultiSortingsModule


def test_tmp_folder(data: LussacData) -> None:
	module = TestMultiSortingsModule(data)
	assert module.tmp_folder == module.data.tmp_folder / "test_multi_sortings_module" / "all"
	assert module.tmp_folder.exists() and module.tmp_folder.is_dir()


# TODO: Test 'extract_waveforms'


class TestMultiSortingsModule(MultiSortingsModule):
	"""
	This is just a test class.
	"""

	__test__ = False

	def __init__(self, data: LussacData) -> None:
		# Create a smaller data object for testing (faster).
		multi_sortings_data = MultiSortingsData(data, data.sortings)
		super().__init__("test_multi_sortings_module", multi_sortings_data, "all")

	@property
	def default_params(self) -> dict[str, Any]:
		return {
			'cat1': {
				'a': -1,
				'b': 2
			},
			'cat2': 3
		}

	def run(self, params: dict):
		pass
