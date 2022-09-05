import pytest
from lussac.core.module import MonoSortingModule
from lussac.core.module_factory import ModuleFactory
from lussac.modules.export_to_phy import ExportToPhy


def test_ModuleFactory() -> None:
	module_factory = ModuleFactory()

	assert 'export_to_phy' in module_factory.module_classes
	assert module_factory.get_module("export_to_phy") == ExportToPhy

	with pytest.raises(ValueError):
		module_factory.get_module("not_a_module")


def test_get_module_member() -> None:
	with pytest.raises(ModuleNotFoundError):
		ModuleFactory._get_module_member("not_a_module")

	with pytest.raises(Exception):
		ModuleFactory._get_module_member("lussac.core.lussac_data")

	with pytest.raises(Exception):
		ModuleFactory._get_module_member("tests.core.test_modulefactory")


def test_is_member_lussac_module() -> None:
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

	def run(self, params: dict):
		pass


class NonAbstractLussacModule2(MonoSortingModule):
	"""
	This class is a 2nd correct Lussac module.
	"""

	def run(self, params: dict):
		pass
