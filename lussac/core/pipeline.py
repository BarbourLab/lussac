import os
import time
from typing import Type
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import spikeinterface.core as si
from lussac.core.lussac_data import LussacData, MonoSortingData, MultiSortingsData
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

		for module_key, module_params in self.data.params['lussac']['pipeline'].items():
			print('\n' + '*'*34)
			print(f"{' ' + module_key + ' ':*^34}")
			print('*' * 34)

			if os.path.exists(f"{self.data.logs_folder}/{module_key}/output"):
				self.data.sortings = self._load_sortings(module_key)
				continue

			module_name = self._get_module_name(module_key)
			module = self.module_factory.get_module(module_name)

			if issubclass(module, MonoSortingModule):
				for category, params in module_params.items():
					print(f"Running category '{category}':")
					self._run_mono_sorting_module(module, module_key, category, params)
			elif issubclass(module, MultiSortingsModule):
				self._run_multi_sortings_module(module, module_key, module_params)
			else:
				raise Exception("Error: Module does not inherit from MonoSortingModule or MultiSortingsModule.")

			self._save_sortings(module_key)

			# Maybe convert sorting objects to Numpy to avoid having a big tree?

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

			print(f"\t- Sorting  {name:<18}", end=' ')
			t1 = time.perf_counter()

			unit_ids = self.get_unit_ids_for_category(category, sorting)
			sub_sorting, other_sorting = self.split_sorting(sorting, unit_ids)

			data = MonoSortingData(self.data, sub_sorting)
			module_instance = module(module_name, data, category)
			params = module_instance.update_params(params)

			sub_sorting = module_instance.run(params)

			self.data.sortings[name] = self.merge_sortings(sub_sorting, other_sorting)

			t2 = time.perf_counter()
			print(f"(Done in {t2-t1:.1f} s)")

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
			print(f"Running category '{category}':")
			t1 = time.perf_counter()

			sub_sortings = {}
			for name, sorting in self.data.sortings.items():
				if 'sortings' in params and name not in params['sortings']:
					continue

				unit_ids = self.get_unit_ids_for_category(category, sorting)
				sub_sortings[name], _ = self.split_sorting(sorting, unit_ids)

			data = MultiSortingsData(self.data, sub_sortings)
			module_instance = module(module_name, data, category)
			params = module_instance.update_params(params)

			sub_sortings = module_instance.run(params)

			for name, sub_sorting in sub_sortings.items():
				if name not in new_sortings:
					new_sortings[name] = sub_sorting
				else:
					new_sortings[name] = si.UnitsAggregationSorting([new_sortings[name], sub_sorting])

			t2 = time.perf_counter()
			print(f"\tDone in {t2-t1:.1f} s")

		self.data.sortings = new_sortings

	def _save_sortings(self, module_name: str) -> None:
		"""
		Saves the current state of the sortings after a module run.

		@param module_name: str
			The module's name/key in the json file.
		"""

		for name, sorting in self.data.sortings.items():
			sorting.save_to_folder(folder=f"{self.data.logs_folder}/{module_name}/output/{name}", verbose=False)

	def _load_sortings(self, module_name: str) -> dict[str, si.BaseSorting]:
		"""
		Loads the sortings from a previous module run.

		@param module_name: str
			The module's name/key in the json file.
		@return sortings: dict[str, si.BaseSorting]
			The loaded sortings.
		"""

		print("Loading sortings from previous run...")
		t1 = time.perf_counter()
		sortings_name = os.listdir(f"{self.data.logs_folder}/{module_name}/output")
		sortings = {name: si.load_extractor(f"{self.data.logs_folder}/{module_name}/output/{name}") for name in sortings_name}
		t2 = time.perf_counter()
		print(f"Done in {t2-t1:.2f} s")

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
		@param sorting: se.SortingExtractor
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
				indices = np.where(units_category == '')[0]
				unit_ids.extend(sorting.unit_ids[indices])
			else:
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

		if len(unit_ids) == sorting.get_num_units():
			return sorting, si.NumpySorting.from_dict({}, sampling_frequency=sorting.get_sampling_frequency())
		if len(unit_ids) == 0:
			return si.NumpySorting.from_dict({}, sampling_frequency=sorting.get_sampling_frequency()), sorting

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

		renamed_unit_ids = [*sorting1.unit_ids, *sorting2.unit_ids]  # TODO: Check for duplicates! unit_ids might not be unique.
		return si.UnitsAggregationSorting([sorting1, sorting2], renamed_unit_ids=renamed_unit_ids)
