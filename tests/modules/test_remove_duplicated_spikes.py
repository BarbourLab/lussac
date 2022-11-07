from lussac.core.lussac_data import MonoSortingData
from lussac.modules.remove_duplicated_spikes import RemoveDuplicatedSpikes


def test_default_params(mono_sorting_data: MonoSortingData) -> None:
	module = RemoveDuplicatedSpikes("test_rds_params", mono_sorting_data, "all")
	assert isinstance(module.default_params, dict)
