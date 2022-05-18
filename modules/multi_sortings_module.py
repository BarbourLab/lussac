from abc import abstractmethod
from core.lussac_data import LussacData
from modules.module import LussacModule


class MultiSortingsModule(LussacModule):
	"""

	"""

	data: LussacData

	@abstractmethod
	def run(self):
		pass
