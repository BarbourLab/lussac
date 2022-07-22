import os
import importlib
import inspect
from typing import Type
from core.module import LussacModule


class ModuleFactory:
	"""
	This is a class to load all the modules in the "modules" folder.
	"""

	module_classes: dict[str, Type[LussacModule]]

	def __init__(self):
		"""
		Loads all the modules in the "modules" folder.
		"""

		self.module_classes = self._load_modules()
		print(self.module_classes)

	@staticmethod
	def _load_modules():
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
	def _is_member_lussac_module(member):
		return inspect.isclass(member) and issubclass(member, LussacModule) and not inspect.isabstract(member)
