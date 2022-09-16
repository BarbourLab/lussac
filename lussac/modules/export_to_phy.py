from pathlib import Path
import spikeinterface.core as si
from spikeinterface.exporters import export_to_phy
from lussac.core.module import MonoSortingModule


class ExportToPhy(MonoSortingModule):
	"""
	Exports the sorting data to phy.
	"""

	def run(self, params: dict) -> si.BaseSorting:
		wvf_extractor = self.extract_waveforms(**params['wvf_extraction'])
		output_folder = self._format_output_path(params['path'])
		Path(output_folder).parent.mkdir(parents=True, exist_ok=True)

		export_to_phy(wvf_extractor, output_folder, **params['export_params'])

		return self.sorting

	def _format_output_path(self, path: str) -> str:
		"""
		Formats the path to the output folder.
		If there is only 1 sorting, it remains unchanged.
		However, if there are multiple sorting, the sorting's name is added.

		@param path: str
			The initial path to the output folder.
		@return output_path: str
			The formatted output path.
		"""

		if self.data.data.num_sortings > 1:
			path = f"{path}/{self.data.name}"

		return path
