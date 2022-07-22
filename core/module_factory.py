import os
import importlib
import inspect
from typing import Any, Type
from core.module import LussacModule


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
		print(self.module_classes)

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
			module = importlib.import_module(f"modules.{module_name}")
			members = inspect.getmembers(module, ModuleFactory._is_member_lussac_module)

			if len(members) == 0:
				raise Exception(f"Error: Couldn't find a module class for module '{module_name}'.")
			if len(members) > 1:
				raise Exception(f"Error: Found multiple module classes for module '{module_name}'.\n{members}")

			modules[module_name] = members[0][1]

		return modules

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
