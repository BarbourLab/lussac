import copy
import os
import pytest
from typing import Any
import numpy as np
from lussac.core.lussac_data import LussacData
from lussac.core.module import MonoSortingModule, MultiSortingsModule
from lussac.core.pipeline import LussacPipeline
import spikeinterface.core as si
from spikeinterface.core.testing import check_sortings_equal
from spikeinterface.curation import CurationSorting


def test_launch(pipeline: LussacPipeline) -> None:
	sortings = pipeline.data.sortings.copy()  # Creates a new dict but not a deepcopy --> same sorting objects.
	params = copy.deepcopy(pipeline.data.params)
	params['lussac']['pipeline'] = {'not_a_module': {'cat': {}}}
	data = LussacData(pipeline.data.recording, sortings, params)

	wrong_pipeline = LussacPipeline(data)
	with pytest.raises(ValueError):
		wrong_pipeline.launch()
	del wrong_pipeline

	light_pipeline = LussacPipeline(data)
	light_pipeline.data.sortings.pop('ks2_cs')
	light_pipeline.data.sortings.pop('ks2_low_thresh')
	light_pipeline.data.sortings.pop('ms3_cs')
	light_pipeline.data.sortings.pop('ms4_cs')
	light_pipeline.data.params['lussac']['pipeline'] = {
		'remove_bad_units': {
			'all': {'firing_rate': {'min': 0.1, 'max': 200}}
		},
		'units_categorization': {
			'all': {'CS': {
				"firing_rate": {
					"min": 0.2,
					"max": 5.0
				},
				"ISI_portion": {
					"range": [8.0, 35.0],
					"max": 0.02
				}
			}}
		},
		'remove_bad_units_2': {
			'CS': {'contamination': {'refractory_period': [2.0, 25.0], 'max': 0.05}}
		},
		'merge_sortings': {
			'CS': {
				'refractory_period': [0.7, 25.0],
				'similarity': {
					'min_similarity': 0.4
				}
			},
			'rest': {
				'refractory_period': [0.17, 0.99],
				'similarity': {
					'min_similarity': 0.4
				}
			}
		}
	}
	light_pipeline.launch()

	assert len(light_pipeline.data.sortings) == 1
	assert not isinstance(light_pipeline.data.sortings['merged_sorting'], si.NpzSortingExtractor)  # The pipeline computed the result.
	computed_sorting = light_pipeline.data.sortings['merged_sorting']
	assert computed_sorting.get_num_units() > 0

	# Check that the sortings have been exported.
	assert os.path.isdir(f"{light_pipeline.data.logs_folder}/remove_bad_units/output/ks2_best")
	assert os.path.isdir(f"{light_pipeline.data.logs_folder}/merge_sortings/output/merged_sorting")

	light_pipeline.launch()
	loaded_sorting = light_pipeline.data.sortings['merged_sorting']
	assert isinstance(loaded_sorting, si.NpzSortingExtractor)  # The pipeline loaded the result.

	print(computed_sorting)
	print(loaded_sorting)

	check_sortings_equal(computed_sorting, loaded_sorting, check_annotations=True, check_properties=True)


def test_run_mono_sorting_module(pipeline: LussacPipeline) -> None:
	n_units = {name: len(pipeline.data.sortings[name].unit_ids) for name in pipeline.data.sortings.keys()}
	n_units['ks2_cs'] -= 1
	pipeline._run_mono_sorting_module(TestMonoSortingModule, "test_mono_starting_module", "all", {})

	for sorting_name in pipeline.data.sortings.keys():
		assert pipeline.data.sortings[sorting_name].get_num_units() == n_units[sorting_name]


def test_run_multi_sortings_module(pipeline: LussacPipeline) -> None:
	pipeline = copy.deepcopy(pipeline)
	pipeline2 = copy.deepcopy(pipeline)

	# Run on all sortings.
	n_units = {name: len(pipeline.data.sortings[name].unit_ids) for name in pipeline.data.sortings.keys()}
	n_units['ks2_cs'] -= 1
	pipeline._run_multi_sortings_module(TestMultiSortingsModule, "test_multi_starting_module", {'all': {}})

	for sorting_name in pipeline.data.sortings.keys():
		assert pipeline.data.sortings[sorting_name].get_num_units() == n_units[sorting_name]

	# Run on a subset of sortings.
	n_units = {name: len(pipeline2.data.sortings[name].unit_ids) for name in pipeline2.data.sortings.keys()}
	n_units['ks2_cs'] -= 1
	pipeline2._run_multi_sortings_module(TestMultiSortingsModule, "test_multi_starting_module", {'all': {'sortings': ['ks2_cs', 'ms3_best', 'ms4_cs']}})

	for sorting_name in pipeline2.data.sortings.keys():
		assert pipeline2.data.sortings[sorting_name].get_num_units() == n_units[sorting_name]

	# TODO: Missing test for aggregation.


def test_get_module_name() -> None:
	assert LussacPipeline._get_module_name("module") == "module"
	assert LussacPipeline._get_module_name("merge_sortings") == "merge_sortings"
	assert LussacPipeline._get_module_name("remove_bad_units_81") == "remove_bad_units"


def test_get_unit_ids_for_category(data: LussacData) -> None:
	unit_ids = np.array([3, 4, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 34, 38, 39, 40, 41, 42, 43, 46, 51], dtype=np.int32)
	categories = np.array(["CS", "CS", "CS", "CS", "MS", "CS", "MS", "MF", "SS", "SS", "SS", "SS", "G",
						   "G", "MF", "CS", "MS", "CS", "SS", "SS", "SS", "SS", "MF", "CS"])

	sorting = data.sortings['ks2_best']
	sorting.set_property("lussac_category", categories, ids=unit_ids, missing_value=None)

	assert np.all(LussacPipeline.get_unit_ids_for_category("CS", sorting) == (3, 4, 5, 7, 11, 34, 39, 51))
	assert np.all(LussacPipeline.get_unit_ids_for_category("MF+G", sorting) == (13, 18, 19, 21, 46))
	assert np.all(LussacPipeline.get_unit_ids_for_category("rest", sorting) ==
				  (0, 1, 2, 6, 8, 10, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 44, 45, 47, 48, 49, 50))
	assert np.all(LussacPipeline.get_unit_ids_for_category("rest+SS", sorting) ==
				  (0, 1, 2, 6, 8, 10, 14, 15, 16, 17, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50))
	assert np.all(LussacPipeline.get_unit_ids_for_category("all", sorting) == sorting.unit_ids)
	assert np.all(LussacPipeline.get_unit_ids_for_category("all+SS+rest", sorting) == sorting.unit_ids)


def test_split_sorting(data: LussacData) -> None:
	sorting = data.sortings['ks2_best']
	unit_ids = np.array([3, 4, 5, 7, 11, 34, 39, 51], dtype=np.int32)

	sorting1, sorting2 = LussacPipeline.split_sorting(sorting, unit_ids)

	assert sorting1.unit_ids.size == 8
	assert sorting2.unit_ids.size == len(sorting.unit_ids) - 8


# TODO: test_merge_sortings

def test_save_load_sortings(pipeline: LussacPipeline) -> None:
	pipeline._save_sortings("test_save_sortings")
	loaded_sortings = pipeline._load_sortings("test_save_sortings")

	assert len(loaded_sortings) == len(pipeline.data.sortings)
	for sorting_name in pipeline.data.sortings.keys():
		assert sorting_name in loaded_sortings
		check_sortings_equal(pipeline.data.sortings[sorting_name], loaded_sortings[sorting_name], check_annotations=True, check_properties=True)


class TestMonoSortingModule(MonoSortingModule):

	__test__ = False

	@property
	def default_params(self) -> dict[str, Any]:
		return {}

	def run(self, params: dict) -> si.BaseSorting:
		if self.sorting.get_annotation('name') == "ks2_cs":
			sorting = CurationSorting(self.sorting)
			sorting.remove_unit(8)
			return sorting.sorting

		return self.sorting


class TestMultiSortingsModule(MultiSortingsModule):

	__test__ = False

	@property
	def default_params(self) -> dict[str, Any]:
		return {}

	def run(self, params: dict) -> dict[str, si.BaseSorting]:
		sorting = CurationSorting(self.sortings['ks2_cs'])
		sorting.remove_unit(5)
		self.sortings['ks2_cs'] = sorting.sorting

		return self.sortings
