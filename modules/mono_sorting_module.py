from abc import abstractmethod
from core.mono_sorting_data import MonoSortingData
from modules.module import LussacModule


class MonoSortingModule(LussacModule):
	"""

	"""

	data: MonoSortingData

	@abstractmethod
	def run(self, params: dict):
		pass
