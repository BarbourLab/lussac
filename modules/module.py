from abc import ABC
from dataclasses import dataclass


@dataclass(slots=True)
class LussacModule(ABC):
	"""
	The abstract module class.
	Every module used in Lussac must inherit from this class.

	Attributes:
		data 		Reference to the data object.
		logs_folder	Path to the folder where to output the logs.
	"""

	data: object
	logs_folder: str
