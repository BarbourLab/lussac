import pathlib
from typing import Any, Sequence
from overrides import override
from lussac.core import MonoSortingModule
import spikeinterface.core as si
from spikeinterface.exporters import export_to_phy


class ExportToPhy(MonoSortingModule):
	"""
	Exports the sorting data to the phy format.
	"""

	export_sortings = False

	@property
	@override
	def default_params(self) -> dict[str, Any]:
		return {
			'wvf_extraction': {
				'ms_before': 1.0,
				'ms_after': 3.0,
				'max_spikes_per_unit': 1_000
			},
			'export_params': {
				'compute_amplitudes': True,
				'compute_pc_features': False,
				'copy_binary': False,
				'template_mode': "average",
				'sparsity': {
					'method': "radius",
					'num_channels': 16,
					'radius_um': 75.0
				},
				'verbose': False
			}
		}

	@override
	def run(self, params: dict[str, Any]) -> si.BaseSorting:
		wvf_extractor = self.extract_waveforms(**params['wvf_extraction'])
		output_folder = pathlib.Path(self._format_output_path(params['path']))

		if params['export_params']['sparsity'] is not None:
			params['export_params']['sparsity'] = si.compute_sparsity(wvf_extractor, **params['export_params']['sparsity'])

		export_to_phy(wvf_extractor, output_folder, **params['export_params'])

		if 'lussac_category' in self.sorting.get_property_keys():
			self.write_tsv_file(output_folder / "lussac_category.tsv", "lussac_category", self.sorting.unit_ids, self.sorting.get_property('lussac_category'))

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

	@staticmethod
	def write_tsv_file(path: pathlib.Path, name: str, keys: Sequence, values: Sequence) -> None:
		"""
		Writes a tsv file with the given keys and values.

		@param path: Path
			Path to the tsv file
		@param name: str
			The name of the column.
		@param keys: Sequence
			The unit ids.
		@param values: Sequence
			The values to write.
		"""
		assert len(keys) == len(values)
		assert path.suffix == ".tsv"

		with open(path, 'w+') as tsv_file:
			tsv_file.write(f"cluster_id\t{name}")
			for key, value in zip(keys, values):
				tsv_file.write(f"\n{key}\t{value}")

			tsv_file.close()
