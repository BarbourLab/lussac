import copy
from pathlib import Path
import pytest
from typing import Any
import numpy as np
from lussac.core.lussac_data import MonoSortingData
from lussac.core.module import MonoSortingModule
from tests.modules.test_remove_bad_units import params


def test_update_params(mono_sorting_data: MonoSortingData) -> None:
	module = TestMonoSortingModule(mono_sorting_data)
	params = {
		'cat1': {
			'a': 1,
			'c': 3
		},
		'cat3': 2
	}
	new_params = module.update_params(params)

	assert new_params == {
		'cat1': {
			'a': 1,
			'b': 2,
			'c': 3
		},
		'cat2': 3,
		'cat3': 2
	}


def test_recording(mono_sorting_module: MonoSortingModule) -> None:
	assert mono_sorting_module.recording == mono_sorting_module.data.recording


def test_sorting(mono_sorting_module: MonoSortingModule) -> None:
	assert mono_sorting_module.sorting == mono_sorting_module.data.sorting


def test_logs_folder(mono_sorting_module: MonoSortingModule) -> None:
	assert mono_sorting_module.logs_folder == f"{mono_sorting_module.data.logs_folder}/test_mono_sorting_module/all/ms3_best"


def test_extract_waveforms(mono_sorting_module: MonoSortingModule) -> None:
	wvf_extractor_1 = mono_sorting_module.extract_waveforms(ms_before=1.5, ms_after=2.0, max_spikes_per_unit=10, overwrite=True)
	wvf_extractor_2 = mono_sorting_module.extract_waveforms(sub_folder="aze", ms_before=1.5, ms_after=2.0, max_spikes_per_unit=10, overwrite=True)

	assert wvf_extractor_1 is not None
	assert wvf_extractor_2 is not None
	assert Path(f"{mono_sorting_module.data.tmp_folder}/test_mono_sorting_module/all/ms3_best/wvf_extractor/waveforms").is_dir()
	assert Path(f"{mono_sorting_module.data.tmp_folder}/test_mono_sorting_module/all/ms3_best/aze/wvf_extractor/waveforms").is_dir()


def test_get_templates(mono_sorting_module: MonoSortingModule) -> None:
	ms_before, ms_after = (2.0, 2.0)
	templates, wvf_extractor, margin = mono_sorting_module.get_templates({'ms_before': ms_before, 'ms_after': ms_after}, filter_band=[300, 6000], return_extractor=True)

	n_units = mono_sorting_module.sorting.get_num_units()
	n_samples = wvf_extractor.nsamples - 2*margin
	n_channels = mono_sorting_module.recording.get_num_channels()

	assert templates is not None
	assert templates.shape == (n_units, n_samples, n_channels)
	assert np.all(wvf_extractor.unit_ids == mono_sorting_module.sorting.unit_ids)

	templates = mono_sorting_module.get_templates({'ms_before': ms_before, 'ms_after': ms_after}, filter_band=[300, 6000], sub_folder="templates2", return_extractor=False)

	assert templates is not None
	assert templates.shape == (n_units, n_samples, n_channels)


def test_get_units_attribute(mono_sorting_data: MonoSortingData) -> None:
	module = TestMonoSortingModule(mono_sorting_data)
	num_units = module.data.sorting.get_num_units()

	frequencies = module.get_units_attribute_arr("firing_rate", params['firing_rate'])
	assert isinstance(frequencies, np.ndarray)
	assert frequencies.shape == (num_units, )
	assert abs(frequencies[0] - 22.978) < 0.01

	contamination = module.get_units_attribute_arr("contamination", params['contamination'])
	assert isinstance(contamination, np.ndarray)
	assert contamination.shape == (num_units, )
	assert contamination[0] >= 1.0

	amplitude = module.get_units_attribute_arr("amplitude", params['amplitude'])
	assert isinstance(amplitude, np.ndarray)
	assert amplitude.shape == (num_units, )

	SNRs = module.get_units_attribute_arr("SNR", params['SNR'])
	assert isinstance(SNRs, np.ndarray)
	assert SNRs.shape == (num_units, )

	amplitude_std = module.get_units_attribute_arr("amplitude_std", params['amplitude_std'])
	assert isinstance(amplitude_std, np.ndarray)
	assert amplitude_std.shape == (num_units, )

	with pytest.raises(ValueError):
		module.get_units_attribute("test", {})


@pytest.fixture(scope="function")
def mono_sorting_module(mono_sorting_data: MonoSortingData) -> MonoSortingModule:
	return TestMonoSortingModule(mono_sorting_data)


class TestMonoSortingModule(MonoSortingModule):
	"""
	This is just a test class.
	"""

	__test__ = False

	def __init__(self, mono_sorting_data: MonoSortingData):
		data = copy.deepcopy(mono_sorting_data)
		data.sorting = data.sorting.select_units([0, 1, 2, 3, 4])
		super().__init__("test_mono_sorting_module", data, "all")

	@property
	def default_params(self) -> dict[str, Any]:
		return {
			'cat1': {
				'a': -1,
				'b': 2
			},
			'cat2': 3
		}

	def run(self, params: dict):
		pass
