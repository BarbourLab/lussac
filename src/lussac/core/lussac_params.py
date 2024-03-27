import json
import pathlib
import platform

import jsmin

import lussac


class LussacParams:

	@staticmethod
	def load_from_string(params: str, params_folder: pathlib.Path | str | None = None):
		"""
		Loads the parameters from a string and returns them as a dict.

		@param params: str
			Lussac's parameters.
		@param params_folder: str
			Path to replace the "$PARAMS_FOLDER".
		"""

		if params_folder is not None:
			params_folder = str(pathlib.Path(params_folder).absolute())
			params = params.replace("$PARAMS_FOLDER", params_folder)
		if platform.system() == "Windows":  # pragma: no cover (OS specific).
			params = params.replace("\\", "\\\\")

		return json.loads(params)

	@staticmethod
	def load_from_json_file(filename: str, params_folder: pathlib.Path | str | None = None) -> dict:
		"""
		Loads the JSON parameters file and returns its content as a dict.

		@param filename: str
			Path to the file containing Lussac's parameters.
		@param params_folder: Path | str | None
			Path to replace the "$PARAMS_FOLDER".
			If None (default), will use the parent folder of the filename.
		@return params: dict
			Lussac's parameters.
		"""

		if params_folder is None:
			params_folder = str(pathlib.Path(filename).parent.absolute())
		else:
			params_folder = str(pathlib.Path(params_folder).absolute())

		with open(filename) as json_file:
			minified = jsmin.jsmin(json_file.read())  # Parses out comments.
			return LussacParams.load_from_string(minified, params_folder)

	@staticmethod
	def load_default_params(name: str, folder: pathlib.Path | str) -> dict:
		"""
		Loads the default parameters from the "params_example" folder.

		@param name: str
			The name of the default params file to load.
		@param folder: str
			Path to the folder where to create the "lussac" folder.
		@return params: dict
			The default parameters.
		"""

		if not name.startswith("params_"):
			name = f"params_{name}"
		if not name.endswith(".json"):
			name = f"{name}.json"

		params_folder = pathlib.Path(lussac.__file__).parent / "params_examples"
		file = params_folder / name

		return LussacParams.load_from_json_file(str(file), folder)
