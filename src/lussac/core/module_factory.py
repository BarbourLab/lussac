import os
import importlib
import inspect
from typing import Any, Type
from lussac.core import LussacModule


class ModuleFactory:
	"""
	This is a class to load all the modules in the "modules" folder.
	"""

	module_classes: dict[str, Type[LussacModule]]

	def __init__(self) -> None:
		"""
		Creates a new ModuleFactory instance.
		Loads all the modules from the "modules" folder.
		"""

		self.module_classes = self._load_modules()

	@staticmethod
	def _load_modules() -> dict[str, Type[LussacModule]]:
		"""
		Loads all the modules from the "modules" folder.

		@return modules: dict[str, Type[LussacModule]]
			All the modules classes.
		"""

		modules = {}

		for module_file in os.listdir(os.path.join(os.path.dirname(__file__), '../modules')):
			if not module_file.endswith('.py') or module_file.startswith('_'):
				continue

			module_name = module_file[:-3]
			modules[module_name] = ModuleFactory._get_module_member(f"lussac.modules.{module_name}")

		return modules

	@staticmethod
	def _get_module_member(module_name: str) -> Type[LussacModule]:
		"""
		Returns the module class from its path.

		@param module_name: str
			Path to the module file as would be used in an import statement.
		@return: module_class: Type[LussacModule]
			The module class (not instantiated!).
		"""

		module = importlib.import_module(module_name)
		members = inspect.getmembers(module, ModuleFactory._is_member_lussac_module)
		members = [member for member in members if member[1].__module__ == module_name]

		if len(members) == 0:
			raise Exception(f"Error: Couldn't find a module class for module '{module_name}'.")
		if len(members) > 1:
			raise Exception(f"Error: Found multiple module classes for module '{module_name}'.\n{members}")

		return members[0][1]

	@staticmethod
	def _is_member_lussac_module(member: Any) -> bool:
		"""
		Checks if a member (from inspect.getmembers) is a LussacModule class.

		@param member: Any
			The member to check.
		@return is_lussac_module: bool
			True if the member is a LussacModule class.
		"""

		return inspect.isclass(member) and issubclass(member, LussacModule) and not inspect.isabstract(member)

	def get_module(self, name: str) -> Type[LussacModule]:
		"""
		Gets a module class by its name.

		@param name: str
			The name of the module.
		@return module: Type[LussacModule]
			The module class.
		"""

		if name not in self.module_classes:
			raise ValueError(f"Error: Module '{name}' not found.\nThe loaded modules are: {list(self.module_classes.keys())}")

		return self.module_classes[name]
