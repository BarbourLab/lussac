import os
from lussac.core.lussac_data import MonoSortingData
from lussac.core.pipeline import LussacPipeline
from lussac.modules.export_to_phy import ExportToPhy


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
			'chunk_duration': '1s',
			'n_jobs': 6
		},
		'export_params': {
			'compute_pc_features': False,
			'compute_amplitudes': False,
			'max_channels_per_template': 4,
			'template_mode': "average",
			'copy_binary': False,
			'chunk_duration': '1s',
			'n_jobs': 6
		}
	}

	pipeline._run_mono_sorting_module(ExportToPhy, "export_to_phy", "all", params)

	assert os.path.exists(f"{folder}/ks2_best/spike_times.npy")
	assert os.path.exists(f"{folder}/ms3_cs/spike_times.npy")
	assert not os.path.exists(f"{folder}/ks2_cs/spike_times.npy")

# TODO: Add test for _format_output_path() for 1 and multiple sortings.
