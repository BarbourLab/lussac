import pytest
import numpy as np
from lussac.core import MonoSortingData, TemplateExtractor


params = {
	'ms_before': 0.5,
	'ms_after': 1.0,
	'max_spikes_per_unit': 400,
	'max_spikes_sparsity': 60
}


@pytest.fixture(scope="module")
def template_extractor(mono_sorting_data: MonoSortingData) -> TemplateExtractor:
	recording = mono_sorting_data.recording
	sorting = mono_sorting_data.sorting
	folder = mono_sorting_data.tmp_folder / "test_template_extractor"

	return TemplateExtractor(recording, sorting, folder, params)


def test_sampling_frequency(template_extractor: TemplateExtractor) -> None:
	assert template_extractor.sampling_frequency == template_extractor.recording.sampling_frequency


def test_unit_ids(template_extractor: TemplateExtractor) -> None:
	assert np.all(template_extractor.unit_ids == template_extractor.sorting.unit_ids)


def test_num_units(template_extractor: TemplateExtractor) -> None:
	assert template_extractor.num_units == template_extractor.sorting.get_num_units()


def test_channel_ids(template_extractor: TemplateExtractor) -> None:
	assert np.all(template_extractor.channel_ids == template_extractor.recording.channel_ids)


def test_num_channels(template_extractor: TemplateExtractor) -> None:
	assert template_extractor.num_channels == template_extractor.recording.get_num_channels()


def test_name(template_extractor: TemplateExtractor) -> None:
	assert template_extractor.name == template_extractor.sorting.get_annotation("name")


def test_nbefore(template_extractor: TemplateExtractor) -> None:
	nbefore = template_extractor.nbefore
	assert nbefore == params['ms_before'] * template_extractor.sampling_frequency * 1e-3

	params2 = params.copy()
	params2['ms_before'] = params['ms_before'] + 1e-5
	template_extractor.set_params(**params2)
	assert template_extractor.nbefore == nbefore + 1  # Should always round up
	template_extractor.set_params(**params)


def test_nafter(template_extractor: TemplateExtractor) -> None:
	nafter = template_extractor.nafter
	assert nafter == params['ms_after'] * template_extractor.sampling_frequency * 1e-3

	params2 = params.copy()
	params2['ms_after'] = params['ms_after'] + 1e-5
	template_extractor.set_params(**params2)
	assert template_extractor.nafter == nafter + 1  # Should always round up
	template_extractor.set_params(**params)


def test_nsamples(template_extractor: TemplateExtractor) -> None:
	assert template_extractor.nsamples == 1 + (params['ms_before'] + params['ms_after']) * template_extractor.sampling_frequency * 1e-3


def test_compute_templates(template_extractor: TemplateExtractor) -> None:
	assert np.isnan(template_extractor._templates[5]).all()
	template_extractor.compute_templates(unit_ids=[5, 9], channel_ids=[14, 19, 3])
	assert not np.isnan(template_extractor._templates[5, :, 14]).any()
	assert np.isnan(template_extractor._templates[5, :, 15]).any()


def test_get_template(template_extractor) -> None:
	assert np.isnan(template_extractor._templates[71]).all()
	template_unscaled = template_extractor.get_template(unit_id=71, channel_ids=np.arange(19, 26))
	assert template_unscaled.shape == (template_extractor.nsamples, 7)
	assert np.all(template_unscaled == template_extractor._templates[71, :, 19:26])
	assert not np.isnan(template_extractor._templates[71, :, 19:26]).any()
	assert np.isnan(template_extractor._templates[71, :, :19]).all()

	template_scaled = template_extractor.get_template(unit_id=71, channel_ids=np.arange(19, 26), return_scaled=True)
	assert 200 < np.max(np.abs(template_scaled)) < 800  # Signal of this Purkinje cell should be around 380 µV.

	template_extractor.get_template(unit_id=71, channel_ids=None)
	assert not np.isnan(template_extractor._templates[71]).any()
