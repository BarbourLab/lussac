from typing import Any
from lussac.core.module import MonoSortingModule
import spikeinterface.core as si


class RemoveDuplicatedSpikes(MonoSortingModule):
	"""
	Removes duplicated spikes from all units.
	Spikes are considered duplicated if they are separated by less than x time samples.
	"""

	@property
	def default_params(self) -> dict[str, Any]:
		return {}

	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		return self.sorting
