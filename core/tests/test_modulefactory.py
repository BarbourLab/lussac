from core.module_factory import ModuleFactory
from modules.export_to_phy import ExportToPhy


def test_ModuleFactory():
	moduleFactory = ModuleFactory()

	assert 'export_to_phy' in moduleFactory.module_classes
	assert moduleFactory.get_module("export_to_phy") == ExportToPhy
