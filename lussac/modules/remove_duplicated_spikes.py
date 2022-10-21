import spikeinterface.core as si
from lussac.core.module import MonoSortingModule


class RemovedDuplicatedSpikes(MonoSortingModule):
	"""
	Removes duplicated spikes from all units.
	Spikes are considered duplicated if they are separated by less than x time samples.
	"""

	def run(self, params: dict) -> si.BaseSorting:
		return self.sorting
