from pathlib import Path
import pytest
from lussac.core.lussac_data import MonoSortingData
from lussac.core.module import MonoSortingModule


def test_recording(mono_sorting_module: MonoSortingModule) -> None:
	assert mono_sorting_module.recording == mono_sorting_module.data.recording


def test_sorting(mono_sorting_module: MonoSortingModule) -> None:
	assert mono_sorting_module.sorting == mono_sorting_module.data.sorting


def test_extract_waveforms(mono_sorting_module: MonoSortingModule) -> None:
	wvf_extractor_1 = mono_sorting_module.extract_waveforms(ms_before=1.5, ms_after=2.0, max_spikes_per_unit=10, overwrite=True)
	wvf_extractor_2 = mono_sorting_module.extract_waveforms(sub_folder="aze", ms_before=1.5, ms_after=2.0, max_spikes_per_unit=10, overwrite=True)

	assert wvf_extractor_1 is not None
	assert wvf_extractor_2 is not None
	assert Path(f"{mono_sorting_module.data.tmp_folder}/test_mono_sorting_data/waveforms/all/ms3_best/waveforms").is_dir()
	assert Path(f"{mono_sorting_module.data.tmp_folder}/test_mono_sorting_data/waveforms/all/ms3_best/aze/waveforms").is_dir()


@pytest.fixture(scope="function")
def mono_sorting_module(mono_sorting_data: MonoSortingData) -> MonoSortingModule:
	return TestMonoSortingModule(mono_sorting_data)


class TestMonoSortingModule(MonoSortingModule):
	"""
	This is just a test class.
	"""

	__test__ = False

	def __init__(self, data: MonoSortingData):
		super().__init__("test_mono_sorting_data", data, "all", "")

	def run(self, params: dict):
		pass
