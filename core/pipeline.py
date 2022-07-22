from dataclasses import dataclass
from core.lussac_data import LussacData
from core.module_factory import ModuleFactory


@dataclass(slots=True)
class LussacPipeline:
	"""

	"""

	data: LussacData
	module_factory: ModuleFactory = ModuleFactory()

	def launch(self) -> None:
		"""
		TODO
		"""

		pass
