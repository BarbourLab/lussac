import pytest
import numpy as np
from lussac.core import MonoSortingData, TemplateExtractor
import spikeinterface.core as si


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

	template_extractor = TemplateExtractor(recording, sorting, folder, None)
	template_extractor.set_params(**params)

	return template_extractor


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


def test_get_template(template_extractor: TemplateExtractor) -> None:
	assert np.isnan(template_extractor._templates[71]).all()
	template_unscaled = template_extractor.get_template(unit_id=71, channel_ids=np.arange(19, 26))
	assert template_unscaled.shape == (template_extractor.nsamples, 7)
	assert np.all(template_unscaled == template_extractor._templates[71, :, 19:26])
	assert not np.isnan(template_extractor._templates[71, :, 19:26]).any()
	assert np.isnan(template_extractor._templates[71, :, :19]).all()

	template_extractor.params['max_spikes_per_unit'] = None
	template_extractor.get_template(38)
	template_extractor.params['max_spikes_per_unit'] = params['max_spikes_per_unit']

	template_scaled = template_extractor.get_template(unit_id=71, channel_ids=np.arange(19, 26), return_scaled=True)
	assert 200 < np.max(np.abs(template_scaled)) < 800  # Signal of this Purkinje cell should be around 380 µV.

	template_extractor.get_template(unit_id=71, channel_ids=None)
	assert not np.isnan(template_extractor._templates[71]).any()


def test_get_templates(template_extractor: TemplateExtractor) -> None:
	assert np.isnan(template_extractor._templates[50:55]).all()
	templates = template_extractor.get_templates(unit_ids=np.arange(50, 53), channel_ids=np.arange(3, 7))
	assert templates.shape == (3, template_extractor.nsamples, 4)
	assert np.all(templates == template_extractor._templates[50:53, :, 3:7])
	assert not np.isnan(template_extractor._templates[50:53, :, 3:7]).any()
	assert np.isnan(template_extractor._templates[50:53, :, :3]).all()
	assert np.isnan(template_extractor._templates[50:53, :, 7:]).all()
	assert np.isnan(template_extractor._templates[53:55]).all()

	templates = template_extractor.get_templates(channel_ids=[1])
	assert templates.shape == (template_extractor.num_units, template_extractor.nsamples, 1)
	assert not np.isnan(template_extractor._templates[:, :, 1]).any()

	template_extractor.get_templates(unit_ids=[27], channel_ids=[1, 2])
	template_extractor.get_templates(unit_ids=[27], channel_ids=[1, 2])  # Test everything already computed
	assert np.isnan(template_extractor._templates[26, :, 2:]).all()


def test_compute_best_channels(template_extractor: TemplateExtractor) -> None:
	assert np.all(template_extractor._best_channels == 0)
	template_extractor.compute_best_channels(unit_ids=[2, 71])
	assert np.sum(template_extractor._best_channels[[2, 71]] == 0) <= 2
	assert np.all(template_extractor._best_channels[71, :3] == (23, 21, 25))

	template_extractor.compute_best_channels(unit_ids=[71])  # Already computed
	assert np.all(template_extractor._best_channels[60] == 0)

	template_extractor.compute_best_channels(unit_ids=None)
	assert np.sum(template_extractor._best_channels == 0) <= template_extractor.num_units
	template_extractor._best_channels[:] = 0  # Reset for test_get_units_best_channels


def test_get_unit_best_channels(template_extractor: TemplateExtractor) -> None:
	assert np.all(template_extractor._best_channels[71] == 0)
	best_channels = template_extractor.get_unit_best_channels(unit_id=71)
	assert best_channels.shape == (template_extractor.num_channels,)
	assert np.all(best_channels == template_extractor._best_channels[71])
	assert np.all(template_extractor._best_channels[:71] == 0)
	assert np.all(template_extractor._best_channels[72:] == 0)
	assert np.all(best_channels[:3] == (23, 21, 25))


def test_get_units_best_channels(template_extractor: TemplateExtractor) -> None:
	assert np.all(template_extractor._best_channels[10:20] == 0)
	best_channels = template_extractor.get_units_best_channels(unit_ids=np.arange(12, 15))
	assert best_channels.shape == (3, template_extractor.num_channels)
	assert np.all(best_channels == template_extractor._best_channels[12:15])
	assert np.sum(template_extractor._best_channels[12:15] == 0) <= 3
	assert np.all(template_extractor._best_channels[10:12] == 0)
	assert np.all(template_extractor._best_channels[15:20] == 0)
	assert best_channels[2, 0] == 5

	best_channels = template_extractor.get_units_best_channels(unit_ids=None)
	assert np.sum(best_channels == 0) <= template_extractor.num_units


def test_empty_units(template_extractor: TemplateExtractor) -> None:
	sorting = si.NumpySorting.from_unit_dict({}, 30_000)  # Make sure it doesn't crash with no units
	te = TemplateExtractor(template_extractor.recording, sorting, template_extractor.folder, template_extractor.params)
	templates = te.get_templates(return_scaled=True)
	assert templates.shape == (0, te.nsamples, te.num_channels)
