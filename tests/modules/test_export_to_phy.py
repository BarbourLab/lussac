import copy
import os
import numpy as np
import pandas as pd
from lussac.core import LussacData, LussacPipeline, MonoSortingData
from lussac.modules import ExportToPhy
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.qualitymetrics as sqm


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = ExportToPhy("test_etp_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_export_multiple_sortings(pipeline: LussacPipeline) -> None:
	folder = "tests/datasets/cerebellar_cortex/lussac/output_phy"
	params = {
		'sortings': ['ks2_best', 'ms3_cs'],
		'path': folder,
		'wvf_extraction': {
			'ms_before': 1.0,
			'ms_after': 2.0,
			'max_spikes_per_unit': 10,
			'sparse': False
		},
		'export_params': {
			'compute_pc_features': False,
			'compute_amplitudes': False,
			'sparsity': {
				'method': "radius",
				'radius_um': 50
			},
			'template_mode': "average",
			'copy_binary': False
		},
		'estimate_contamination': {
			'all': (0.3, 0.9),
			'CS': (1.0, 25.0)
		}
	}

	pipeline._run_mono_sorting_module(ExportToPhy, "export_to_phy", "all", params)

	assert os.path.exists(f"{folder}/ks2_best/spike_times.npy")
	assert os.path.exists(f"{folder}/ms3_cs/spike_times.npy")
	assert not os.path.exists(f"{folder}/ks2_cs/spike_times.npy")

	sorting_saved = se.PhySortingExtractor(f"{folder}/ms3_cs")
	sorting = pipeline.data.sortings['ms3_cs']
	assert np.all(sorting.to_spike_vector() == sorting_saved.to_spike_vector())

	analyzer = si.create_sorting_analyzer(sorting, pipeline.data.recording)
	contamination = sqm.compute_refrac_period_violations(analyzer, refractory_period_ms=0.9, censored_period_ms=0.3)[0]
	saved_contamination = dict(zip(sorting_saved.unit_ids, sorting_saved.get_property("lussac_cont (%)")))
	assert np.all(list(contamination.keys()) == list(saved_contamination.keys()))
	for unit_id in sorting.unit_ids:
		assert np.isclose(contamination[unit_id], 0.01*saved_contamination[unit_id], atol=1e-5)


def test_empty_sorting(data: LussacData) -> None:
	mono_sorting_data = MonoSortingData(data, si.NumpySorting.from_unit_dict({}, sampling_frequency=30000))
	module = ExportToPhy("test_etp_empty", mono_sorting_data, "all")
	module.run(module.default_params)


def test_format_output_path(mono_sorting_data: MonoSortingData) -> None:
	module = ExportToPhy("test_etp_format_path", mono_sorting_data, "all")
	assert module._format_output_path("test") == "test/ms3_best"

	module = copy.deepcopy(module)
	module.data.data.sortings = {'ms3_best': module.sorting}
	assert module._format_output_path("test") == "test"


def test_estimate_units_contamination(mono_sorting_data: MonoSortingData) -> None:
	module = ExportToPhy("test_etp_contamination", mono_sorting_data, "all")

	assert 'lussac_category' not in module.sorting.get_property_keys()
	estimated_contamination = module._estimate_units_contamination({'CS': (1.0, 25.0)})
	assert len(estimated_contamination) == 0
	estimated_contamination = module._estimate_units_contamination({'all': (0.3, 0.9)})
	assert len(estimated_contamination) == mono_sorting_data.sorting.get_num_units()


def test_write_tsv_file(mono_sorting_data: MonoSortingData) -> None:
	file_path = mono_sorting_data.tmp_folder / "test.tsv"
	assert not file_path.exists()

	ExportToPhy.write_tsv_file(file_path, "test", [0, 1, 3], ['a', 'b', 'c'])
	assert file_path.exists()

	df = pd.read_csv(file_path, sep='\t')
	assert np.all(np.array(df.columns) == ("cluster_id", "test"))
	assert np.all(np.array(df['cluster_id']) == (0, 1, 3))
	assert np.all(np.array(df['test']) == ('a', 'b', 'c'))
