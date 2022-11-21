import os
from lussac.core.lussac_data import MonoSortingData
from lussac.core.pipeline import LussacPipeline
from lussac.modules.export_to_sigui import ExportToSIGUI


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = ExportToSIGUI("test_ets_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)


def test_export_multiple_sortings(pipeline: LussacPipeline) -> None:
	folder = "tests/datasets/cerebellar_cortex/lussac/output_sigui"
	params = {
		'sortings': ['ks2_best', 'ms3_cs'],
		'path': folder,
		'wvf_extraction': {
			'ms_before': 1.0,
			'ms_after': 2.0,
			'max_spikes_per_unit': 10,
			'allow_unfiltered': True,
			'chunk_duration': '1s',
			'n_jobs': 6
		},
		'spike_amplitudes': False,
		'principal_components': False
	}

	pipeline._run_mono_sorting_module(ExportToSIGUI, "export_to_sigui", "all", params)

	assert os.path.exists(f"{folder}/ks2_best/waveforms")
	assert os.path.exists(f"{folder}/ms3_cs/waveforms")
	assert not os.path.exists(f"{folder}/ks2_cs/waveforms")

# TODO: Add test for _format_output_path() for 1 and multiple sortings.
