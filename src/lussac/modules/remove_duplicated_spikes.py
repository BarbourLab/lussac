from typing import Any
from overrides import override
from lussac.core import MonoSortingModule
import spikeinterface.core as si
import spikeinterface.curation as scur


class RemoveDuplicatedSpikes(MonoSortingModule):
	"""
	Removes duplicated spikes from all units.
	Spikes are considered duplicated if they are separated by less than x time samples.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'censored_period': 0.3,
			'method': "keep_first_iterative"
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		return scur.remove_duplicated_spikes(self.sorting, params['censored_period'], params['method'])
