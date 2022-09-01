from typing import ClassVar
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import spikeinterface.core as si
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

	categories: ClassVar[dict[str, dict[str, dict]]] = {}

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

	@staticmethod
	def get_unit_ids_for_category(category: str, sorting: si.BaseSorting) -> npt.NDArray[np.integer]:
		"""
		Gets all the unit ids for a given category.

		@param category: str
			The category to get the unit ids for.
		@param sorting: se.SortingExtractor
			The sorting to get the unit ids from.
		@return unit_ids: list[int]
			The unit ids for the given category.
		"""

		unit_ids = []
		units_category = sorting.get_property("lussac_category")

		categories = category.split('+')
		for cat in categories:
			if cat == "all":
				unit_ids.extend(sorting.unit_ids)
			elif cat == "rest":
				indices = np.where(units_category == '')[0]
				unit_ids.extend(sorting.unit_ids[indices])
			else:
				indices = np.where(units_category == cat)[0]
				unit_ids.extend(sorting.unit_ids[indices])

		return np.sort(np.unique(unit_ids))
