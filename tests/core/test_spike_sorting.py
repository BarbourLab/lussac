import importlib.util
import pytest
from lussac.core import LussacData, LussacSpikeSorter
import spikeinterface.core as si


@pytest.mark.skipif(importlib.util.find_spec("hdbscan") is None, reason="Testing the spike sorting requires hdbscan.")
def test_spike_sorting(data: LussacData) -> None:
	recording = data.recording.frame_slice(9_000_000, 18_000_000)  # 5-min recording to go faster, don't care about result.
	params = {
		'sorter': "spykingcircus2",
		'preprocessing': {
			'filter': {'band': [150., 6000.], 'filter_order': 2, 'ftype': "bessel"},
			'common_reference': {'operator': "median"}
		},
		'sorter_params': {
			'output_folder': f"{data.tmp_folder}/sc2_ss_test"
		}
	}

	spike_sorter = LussacSpikeSorter(recording)
	sorting = spike_sorter.launch(params)

	assert isinstance(sorting, si.BaseSorting)
	assert sorting.get_num_units() > 0
