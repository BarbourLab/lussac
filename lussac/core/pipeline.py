from dataclasses import dataclass
from lussac.core.lussac_data import LussacData
from lussac.core.module import MonoSortingModule, MultiSortingsModule
from lussac.core.module_factory import ModuleFactory


@dataclass(slots=True)
class LussacPipeline:
	"""
	The pipeline object for Lussac.

	Attributes:
		data			Reference to the data object.
		module_factory	Object to load and run all the modules.
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
				run_module = self._run_mono_sorting_module
			elif isinstance(module, MultiSortingsModule):
				run_module = self._run_multi_sortings_module
			else:
				raise Exception("Error: Module does not inherit from MonoSortingModule or MultiSortingsModule.")

			for category, params in value.items():
				run_module(module, category)

	def _run_mono_sorting_module(self, module: MonoSortingModule, category: str) -> None:
		"""
		Launches a mono-sorting module for a category on all sortings.

		@param module: MonoSortingModule
			The module class to use.
		@param category: str
			TODO
		"""

		pass

	def _run_multi_sortings_module(self, module: MultiSortingsModule, category: str) -> None:
		"""
		Launches a multi-sorting module for a category.

		@param module: MultiSortingsModule
			The module class to use.
		@param category: str
			TODO
		"""

		pass

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
