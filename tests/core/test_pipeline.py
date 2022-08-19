from lussac.core.pipeline import LussacPipeline


def test_get_module_name() -> None:
	assert LussacPipeline._get_module_name("module") == "module"
	assert LussacPipeline._get_module_name("merge_sortings") == "merge_sortings"
	assert LussacPipeline._get_module_name("remove_bad_units_81") == "remove_bad_units"

