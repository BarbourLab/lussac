import copy
import os
from lussac.core import LussacData, LussacPipeline, MonoSortingData
from lussac.modules import ExportToSIGUI


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = ExportToSIGUI("test_ets_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_export_multiple_sortings(data: LussacData) -> None:
	data = data.clone()
	data.recording = data.recording.frame_slice(0, 1_000_000)
	data.sortings = {name: data.sortings[name].frame_slice(0, 1_000_000) for name in data.sortings.keys() if name in ["ks2_cs", "ms3_cs", "ks2_best"]}
	pipeline = LussacPipeline(data)

	folder = "tests/datasets/cerebellar_cortex/lussac/output_sigui"
	params = {
		'sortings': ['ks2_best', 'ms3_cs'],
		'path': folder,
		'wvf_extraction': {
			'ms_before': 1.0,
			'ms_after': 2.0,
			'max_spikes_per_unit': 10,
			'allow_unfiltered': True
		},
		'spike_amplitudes': {},
		'principal_components': {'n_components': 3}
	}

	pipeline._run_mono_sorting_module(ExportToSIGUI, "export_to_sigui", "all", params)

	assert os.path.exists(f"{folder}/ks2_best/extensions")
	assert os.path.exists(f"{folder}/ms3_cs/extensions")
	assert not os.path.exists(f"{folder}/ks2_cs/extensions")


def test_format_output_path(mono_sorting_data: MonoSortingData) -> None:
	module = ExportToSIGUI("test_etp_format_path", mono_sorting_data, "all")
	assert module._format_output_path("test") == "test/ms3_best"

	module = copy.deepcopy(module)
	module.data.data.sortings = {'ms3_best': module.sorting}
	assert module._format_output_path("test") == "test"
