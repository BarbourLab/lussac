import numpy as np
from lussac.core import MonoSortingData
from lussac.modules import RemoveDuplicatedSpikes
import spikeinterface.core as si


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = RemoveDuplicatedSpikes("test_rds_params", mono_sorting_data, "all")
	n_spikes = [len(module.sorting.get_unit_spike_train(unit_id)) for unit_id in module.sorting.unit_ids]
	assert isinstance(module.default_params, dict)

	sorting = module.run({'censored_period': 0.3, 'method': "random"})
	assert np.all([len(sorting.get_unit_spike_train(unit_id)) <= n_spikes[i] for i, unit_id in enumerate(sorting.unit_ids)])

	# Test with known result
	sorting = si.NumpySorting.from_dict({0: np.array([200, 850, 851, 852, 936, 2587]), 1: np.array([250, 853, 1287])}, module.sampling_f)
	data = MonoSortingData(mono_sorting_data.data, sorting)
	module = RemoveDuplicatedSpikes("test_rds_known", data, "all")
	sorting = module.run({'censored_period': 0.3, 'method': "random"})

	assert len(sorting.get_unit_spike_train(0)) == 4
	assert len(sorting.get_unit_spike_train(1)) == 3
