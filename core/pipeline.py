from dataclasses import dataclass
from core.lussac_data import LussacData
from core.module import MonoSortingModule, MultiSortingsModule
from core.module_factory import ModuleFactory


@dataclass(slots=True)
class LussacPipeline:
	"""

	"""

	data: LussacData
	module_factory: ModuleFactory = ModuleFactory()

	def launch(self) -> None:
		"""
		Launches the Lussac's pipeline.
		"""

		for key, value in self.data.params['lussac']['pipeline'].items():
			module_name = self._get_module_name(key)
			module = self.module_factory.get_module(module_name)

			if isinstance(module, MonoSortingModule):
				pass
			elif isinstance(module, MultiSortingsModule):
				pass
			else:
				raise Exception("Error: Module does not inherit from MonoSortingModule or MultiSortingsModule.")

	@staticmethod
	def _get_module_name(name: str) -> str:
		"""
		Gets the module name from the params dictionary key.
		Since all keys in a dict need to be different, this allows to use the same
		module multiple times by adding _{number} to the key.

		@param name: str
			Key from the params dictionary.
		@return module_name: str
			The module's name.
		"""

		if (split := name.split('_'))[-1].isnumeric():
			name = '_'.join(split[:-1])

		return name
