import spikeinterface.core as si
from lussac.core.module import MonoSortingModule


class RemoveBadUnits(MonoSortingModule):
	"""

	"""

	def run(self, params: dict) -> si.BaseSorting:
		raise NotImplementedError()
