from typing import Any
from overrides import override
from lussac.core.module import MonoSortingModule
import spikeinterface.core as si
from spikeinterface.exporters import export_to_phy


class ExportToPhy(MonoSortingModule):
	"""
	Exports the sorting data to the phy format.
	"""

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'wvf_extraction': {
				'ms_before': 1.0,
				'ms_after': 3.0,
				'max_spikes_per_unit': 1000
			},
			'export_params': {
				'compute_amplitudes': True,
				'compute_pc_features': False,
				'copy_binary': False,
				'template_mode': "average",
				'verbose': False,
				'chunk_duration': '1s',
				'n_jobs': 6
			}
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		wvf_extractor = self.extract_waveforms(**params['wvf_extraction'])
		output_folder = self._format_output_path(params['path'])

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
