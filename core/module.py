from abc import ABC, abstractmethod
from dataclasses import dataclass
from core.lussac_data import LussacData, MonoSortingData


@dataclass(slots=True)
class LussacModule(ABC):
	"""
	The abstract Module class.
	Every module used in Lussac must inherit from this class.

	Attributes:
		data 		Reference to the data object.
		logs_folder	Path to the folder where to output the logs.
	"""

	data: object
	logs_folder: str


class MonoSortingModule(LussacModule):
	"""
	The abstract mono-sorting module class.
	This is for modules that don't work on multiple sortings at once.

	Attributes:
		data		Reference to the mono-sorting data object.
		logs_folder	Path to the folder where to output the logs.
	"""

	data: MonoSortingData

	@abstractmethod
	def run(self, params: dict):
		pass


class MultiSortingsModule(LussacModule):
	"""
	The abstract multi-sorting module class.
	This is for modules that work on multiple sortings at once.

	Attributes:
		data		Reference to Lussac data object.
		logs_folder	Path to the folder where to output the logs.
	"""

	data: LussacData

	@abstractmethod
	def run(self):
		pass
