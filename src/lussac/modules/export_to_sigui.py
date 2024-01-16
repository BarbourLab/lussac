from typing import Any
from overrides import override
from lussac.core import MonoSortingModule
import spikeinterface.core as si
import spikeinterface.postprocessing as spost


class ExportToSIGUI(MonoSortingModule):
	"""
	Exports the sorting data as a waveform extractor (for SpikeInterface GUI).
	"""

	export_sortings = False

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'wvf_extraction': {
				'ms_before': 1.0,
				'ms_after': 3.0,
				'max_spikes_per_unit': 1000,
				'return_scaled': True,
				'allow_unfiltered': True
			},
			'spike_amplitudes': {

			},
			'principal_components': False
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		path = self._format_output_path(params['path'])
		wvf_extractor = si.extract_waveforms(self.recording, self.sorting, path, **params['wvf_extraction'])

		if isinstance(params['spike_amplitudes'], dict):
			spost.compute_spike_amplitudes(wvf_extractor, **params['spike_amplitudes'])
		if isinstance(params['principal_components'], dict):
			spost.compute_principal_components(wvf_extractor, **params['principal_components'])

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
