import pytest
from core.module import MonoSortingModule
from core.module_factory import ModuleFactory
from modules.export_to_phy import ExportToPhy


def test_ModuleFactory():
	moduleFactory = ModuleFactory()

	assert 'export_to_phy' in moduleFactory.module_classes
	assert moduleFactory.get_module("export_to_phy") == ExportToPhy

	with pytest.raises(ValueError) as err:
		moduleFactory.get_module("not_a_module")


def test_is_member_lussac_module():
	assert ModuleFactory._is_member_lussac_module(test_ModuleFactory) is False
	assert ModuleFactory._is_member_lussac_module(MonoSortingModule) is False
	assert ModuleFactory._is_member_lussac_module(IsNotLussacModule) is False
	assert ModuleFactory._is_member_lussac_module(AbstractLussacModule) is False
	assert ModuleFactory._is_member_lussac_module(NonAbstractLussacModule)
	assert ModuleFactory._is_member_lussac_module(ExportToPhy)
	assert ModuleFactory._is_member_lussac_module(ExportToPhy.run) is False


class IsNotLussacModule:
	"""
	This class is not a Lussac module.
	"""

	pass


class AbstractLussacModule(MonoSortingModule):
	"""
	This class is a Lussac module, but is still abstract thus it can't be used.
	"""

	pass


class NonAbstractLussacModule(MonoSortingModule):
	"""
	This class is a correct Lussac module.
	"""

	def run(self, params: dict):  # pragma: no cover
		pass
