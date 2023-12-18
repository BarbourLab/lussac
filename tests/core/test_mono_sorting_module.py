import pytest
from typing import Any
import numpy as np
from lussac.core import MonoSortingData, MonoSortingModule


params = {
	"firing_rate": {
		"min": 0.5
	},
	"contamination": {
		"refractory_period": [0.5, 1.0],
		"max": 0.25
	},
	"amplitude": {
		"wvf_extraction": {
			"ms_before": 1.0,
			"ms_after": 1.0,
			"max_spikes_per_unit": 10
		},
		"filter": [200, 5000],
		"min": 20
	},
	"SNR": {
		"wvf_extraction": {
			"ms_before": 1.0,
			"ms_after": 1.0,
			"max_spikes_per_unit": 10
		},
		"filter": [300, 6000],
		"min": 1.2
	},
	"sd_ratio": {
		"wvf_extraction": {
			"ms_before": 1.0,
			"ms_after": 1.0,
		},
		"filter": [200, 5000],
		"max": 2.0
	}
}


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
	assert mono_sorting_module.logs_folder == mono_sorting_module.data.logs_folder / "test_mono_sorting_module" / "all" / "ms3_best"
	assert mono_sorting_module.logs_folder.exists() and mono_sorting_module.logs_folder.is_dir()


def test_tmp_folder(mono_sorting_module: MonoSortingModule) -> None:
	assert mono_sorting_module.tmp_folder == mono_sorting_module.data.tmp_folder / "test_mono_sorting_module" / "all" / "ms3_best"
	assert mono_sorting_module.tmp_folder.exists() and mono_sorting_module.tmp_folder.is_dir()


def test_extract_waveforms(mono_sorting_module: MonoSortingModule) -> None:
	wvf_extractor_1 = mono_sorting_module.extract_waveforms(ms_before=1.5, ms_after=2.0, max_spikes_per_unit=10, overwrite=True)
	wvf_extractor_2 = mono_sorting_module.extract_waveforms(sub_folder="aze", ms_before=1.5, ms_after=2.0, max_spikes_per_unit=10, overwrite=True)
	tmp_folder = mono_sorting_module.data.tmp_folder

	assert wvf_extractor_1 is not None
	assert wvf_extractor_2 is not None
	assert (tmp_folder / "test_mono_sorting_module" / "all" / "ms3_best" / "wvf_extractor" / "waveforms").is_dir()
	assert (tmp_folder / "test_mono_sorting_module" / "all" / "ms3_best" / "aze" / "waveforms").is_dir()


def test_get_templates(mono_sorting_module: MonoSortingModule) -> None:
	ms_before, ms_after = (2.0, 2.0)
	templates, wvf_extractor, margin = mono_sorting_module.get_templates({'ms_before': ms_before, 'ms_after': ms_after, 'max_spikes_per_unit': 10},
																		 filter_band=[300, 6000], return_extractor=True)

	n_units = mono_sorting_module.sorting.get_num_units()
	n_samples = wvf_extractor.nsamples - 2*margin
	n_channels = mono_sorting_module.recording.get_num_channels()

	assert templates is not None
	assert templates.shape == (n_units, n_samples, n_channels)
	assert np.all(wvf_extractor.unit_ids == mono_sorting_module.sorting.unit_ids)

	templates = mono_sorting_module.get_templates({'ms_before': ms_before, 'ms_after': ms_after, 'max_spikes_per_unit': 10},
												  filter_band=[300, 6000], sub_folder="templates2", return_extractor=False)

	assert templates is not None
	assert templates.shape == (n_units, n_samples, n_channels)


def test_get_units_attribute(mono_sorting_data: MonoSortingData) -> None:
	mono_sorting_data = MonoSortingData(mono_sorting_data.data, mono_sorting_data.sorting.select_units([0, 1, 2, 3, 4]))

	module = TestMonoSortingModule(mono_sorting_data)
	num_units = module.data.sorting.get_num_units()

	frequencies = module.get_units_attribute_arr("firing_rate", params['firing_rate'])
	assert isinstance(frequencies, np.ndarray)
	assert frequencies.shape == (num_units, )
	assert abs(frequencies[0] - 22.978) < 0.01

	contaminations = module.get_units_attribute_arr("contamination", params['contamination'])
	assert isinstance(contaminations, np.ndarray)
	assert contaminations.shape == (num_units, )
	assert contaminations[0] >= 1.0

	amplitude = module.get_units_attribute_arr("amplitude", params['amplitude'])
	assert isinstance(amplitude, np.ndarray)
	assert amplitude.shape == (num_units, )

	SNRs = module.get_units_attribute_arr("SNR", params['SNR'])
	assert isinstance(SNRs, np.ndarray)
	assert SNRs.shape == (num_units, )

	sd_ratio = module.get_units_attribute_arr("sd_ratio", params['sd_ratio'])
	assert isinstance(sd_ratio, np.ndarray)
	assert sd_ratio.shape == (num_units, )

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
		# Create a smaller data object for testing (faster).
		data = MonoSortingData(mono_sorting_data.data, mono_sorting_data.sorting.select_units([0, 1, 2, 3, 4]))
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
