import copy
from dataclasses import dataclass
import glob
import logging
import os
import pathlib
import time
from typing import Type
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from lussac.core import LussacData, MonoSortingData, MultiSortingsData, MonoSortingModule, MultiSortingsModule, ModuleFactory
import spikeinterface.core as si


@dataclass(slots=True)
class LussacPipeline:
	"""
	The pipeline object for Lussac.

	Attributes:
		data			Reference to the data object.
		module_factory	Object to load and run all the modules.
	"""

	data: LussacData
	module_factory: ModuleFactory

	def __init__(self, data: LussacData) -> None:
		"""
		Creates a new LussacPipeline instance.

		@param data: LussacData
			The data object.
		"""

		self.data = data
		self.module_factory = ModuleFactory()

	def launch(self) -> None:
		"""
		Launches the Lussac's pipeline.
		"""

		for module_key, module_params in self.data.params['lussac']['pipeline'].items():
			logging.info('\n\n' + ('*'*34) + '\n')
			logging.info(f"{' ' + module_key + ' ':*^34}\n")
			logging.info(('*'*34) + '\n')

			if os.path.exists(f"{self.data.logs_folder}/{module_key}/sorting"):
				self.data.sortings = self._load_sortings(module_key)
				continue

			module_name = self._get_module_name(module_key)
			module = self.module_factory.get_module(module_name)

			if issubclass(module, MonoSortingModule):
				for category, params in module_params.items():
					logging.info(f"Running category '{category}':\n")
					self._run_mono_sorting_module(module, module_key, category, params)
			elif issubclass(module, MultiSortingsModule):
				self._run_multi_sortings_module(module, module_key, module_params)
			else:  # pragma: no cover (unreachable code)
				raise Exception("Error: Module does not inherit from MonoSortingModule or MultiSortingsModule.")

			self._save_sortings(module_key)

	def _run_mono_sorting_module(self, module: Type[MonoSortingModule], module_name: str, category: str, params: dict) -> None:
		"""
		Launches a mono-sorting module for a category on all sortings.

		@param module: MonoSortingModule
			The module class to use.
		@param module_name: str
			The module's name/key in the json file.
		@param category: str
			Run the module only on units from this category.
		@param params: dict
			The parameters for the module.
		"""

		for name, sorting in self.data.sortings.items():
			if 'sortings' in params and name not in params['sortings']:
				continue

			logging.info(f"\t- Sorting  {name:<18}")
			t1 = time.perf_counter()

			unit_ids = self.get_unit_ids_for_category(category, sorting)
			sub_sorting, other_sorting = self.split_sorting(sorting, unit_ids)

			data = MonoSortingData(self.data, sub_sorting)
			module_instance = module(module_name, data, category)
			params0 = copy.deepcopy(params)
			params0 = module_instance.update_params(params0)
			if 'sortings' in params0:
				del params0['sortings']

			sub_sorting = module_instance.run(params0)

			self.data.sortings[name] = self.merge_sortings(sub_sorting, other_sorting)

			t2 = time.perf_counter()
			logging.info(f" (Done in {t2-t1:.1f} s)\n")

	def _run_multi_sortings_module(self, module: Type[MultiSortingsModule], module_name: str, module_params: dict) -> None:
		"""
		Launches a multi-sorting module for a category.

		@param module: MultiSortingsModule
			The module class to use.
		@param module_name: str
			The module's name/key in the json file.
		@param module_params: dict
			The parameters for the module.
		"""

		new_sortings = {}
		for category, params in module_params.items():
			logging.info(f"Running category '{category}':\n")
			t1 = time.perf_counter()

			sub_sortings = {}
			for name, sorting in self.data.sortings.items():
				if 'sortings' in params and name not in params['sortings']:
					continue

				unit_ids = self.get_unit_ids_for_category(category, sorting)
				sub_sortings[name], _ = self.split_sorting(sorting, unit_ids)

			data = MultiSortingsData(self.data, sub_sortings)
			module_instance = module(module_name, data, category)
			params = copy.deepcopy(params)
			params = module_instance.update_params(params)

			sub_sortings = module_instance.run(params)

			for name, sub_sorting in sub_sortings.items():
				if name not in new_sortings:
					new_sortings[name] = sub_sorting
				else:
					new_sortings[name] = si.UnitsAggregationSorting([new_sortings[name], sub_sorting])

			t2 = time.perf_counter()
			logging.info(f"\tDone in {t2-t1:.1f} s\n")

		self.data.sortings = new_sortings

	def _save_sortings(self, module_name: str) -> None:
		"""
		Saves the current state of the sortings after a module run.

		@param module_name: str
			The module's name/key in the json file.
		"""

		for name, sorting in self.data.sortings.items():
			path = f"{self.data.logs_folder}/{module_name}/sorting/{name}.pkl"
			# sorting.dump_to_pickle(file_path=path, include_properties=True, relative_to=self.data.logs_folder)
			sorting.dump_to_pickle(file_path=path, include_properties=True)
			# TODO: Make relative paths work with pickle in SI.

	def _load_sortings(self, module_name: str) -> dict[str, si.BaseSorting]:
		"""
		Loads the sortings from a previous module run.

		@param module_name: str
			The module's name/key in the json file.
		@return sortings: dict[str, si.BaseSorting]
			The loaded sortings.
		"""

		logging.info("Loading sortings from previous run...\n")
		sortings_path = glob.glob(f"{self.data.logs_folder}/{module_name}/sorting/*.pkl")
		# sortings = {pathlib.Path(path).stem: si.load_extractor(path, base_folder=self.data.logs_folder) for path in tqdm(sortings_path)}
		sortings = {pathlib.Path(path).stem: si.load_extractor(path) for path in tqdm(sortings_path)}

		return sortings

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
		@param sorting: BaseSorting
			The sorting to get the unit ids from.
		@return unit_ids: np.ndarray[int]
			The unit ids for the given category.
		"""

		unit_ids = []
		units_category = sorting.get_property("lussac_category")

		categories = category.split('+')
		for cat in categories:
			if cat == "all":
				unit_ids.extend(sorting.unit_ids)
			elif cat == "rest":
				indices = np.where((units_category == '') | (units_category == None))[0]
				unit_ids.extend(sorting.unit_ids[indices])
			elif units_category is not None and len(units_category) > 0:
				indices = np.where(units_category == cat)[0]
				unit_ids.extend(sorting.unit_ids[indices])

		return np.sort(np.unique(unit_ids))

	@staticmethod
	def split_sorting(sorting: si.BaseSorting, unit_ids: npt.ArrayLike) -> tuple[si.BaseSorting, si.BaseSorting]:
		"""
		Splits a sorting into two based on the given unit ids.

		@param sorting: si.BaseSorting
			The sorting to split.
		@param unit_ids: ArrayLike[int]
			The unit ids of the first sorting.
		@return split_sortings: tuple[si.BaseSorting, si.BaseSorting]
			The split sortings.
		"""

		other_unit_ids = [unit_id for unit_id in sorting.get_unit_ids() if unit_id not in unit_ids]
		sorting1 = sorting.select_units(unit_ids)
		sorting2 = sorting.select_units(other_unit_ids)

		return sorting1, sorting2

	@staticmethod
	def merge_sortings(sorting1: si.BaseSorting, sorting2: si.BaseSorting) -> si.BaseSorting:
		"""
		Merges two split sortings into one.

		@param sorting1: si.BaseSorting
			The first sorting.
		@param sorting2: si.BaseSorting
			The second sorting.
		@return merged_sorting: si.BaseSorting
			The merged sorting.
		"""

		if sorting2.get_num_units() == 0:
			return sorting1
		if sorting1.get_num_units() == 0:
			return sorting2

		# Check for unit in sorting2 that has the same id as a unit in sorting1.
		rename_units = {}
		renamed = False
		for unit_id in sorting2.unit_ids:
			if unit_id in sorting1.unit_ids:
				rename_units[unit_id] = np.max([*sorting1.unit_ids, *sorting2.unit_ids, *rename_units.values()]) + 1
				renamed = True
			else:
				rename_units[unit_id] = unit_id

		if renamed:
			sorting2 = sorting2.select_units(list(rename_units.keys()), list(rename_units.values()))

		return si.UnitsAggregationSorting([sorting1, sorting2], renamed_unit_ids=[*sorting1.unit_ids, *sorting2.unit_ids])
