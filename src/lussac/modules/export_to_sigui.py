from typing import Any
from overrides import override
from lussac.core import MonoSortingModule
import spikeinterface.core as si
import spikeinterface.postprocessing as spost


class ExportToSIGUI(MonoSortingModule):
	"""
	Exports the sorting data as a sorting analyzer (for SpikeInterface GUI).
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
				'return_scaled': True
			},
			'spike_amplitudes': {

			},
			'principal_components': False
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		path = self._format_output_path(params['path'])
		analyzer = si.create_sorting_analyzer(self.sorting, self.recording, format="binary_folder", folder=path, return_scaled=params['wvf_extraction']['return_scaled'])

		analyzer_params = {
			'random_spikes': {'max_spikes_per_unit': params['wvf_extraction']['max_spikes_per_unit']},
			'waveforms': {'ms_before': params['wvf_extraction']['ms_before'], 'ms_after': params['wvf_extraction']['ms_after']},
			'templates': {'operators': ["average"]}
		}
		if isinstance(params['spike_amplitudes'], dict):
			analyzer_params['spike_amplitudes'] = params['spike_amplitudes']
		if isinstance(params['principal_components'], dict):
			analyzer_params['principal_components'] = params['principal_components']

		analyzer.compute(analyzer_params)

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
