import pathlib
from typing import Any, Sequence
import pandas as pd
import numpy as np
from overrides import override
from lussac.core import LussacPipeline, MonoSortingModule
import spikeinterface.core as si
from spikeinterface.exporters import export_to_phy
import spikeinterface.qualitymetrics as sqm


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
				'max_spikes_per_unit': 1_000,
				'sparse': False
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
		if self.sorting.get_num_units() == 0:  # Export crashes if the sorting contains no units.
			return self.sorting

		wvf_extractor = self.extract_waveforms(**params['wvf_extraction'])
		output_folder = pathlib.Path(self._format_output_path(params['path']))

		if 'sparsity' in params['export_params'] and params['export_params']['sparsity'] is not None:
			params['export_params']['sparsity'] = si.compute_sparsity(wvf_extractor, **params['export_params']['sparsity'])

		export_to_phy(wvf_extractor, output_folder, **params['export_params'])
		new_unit_ids = pd.read_csv(output_folder / "cluster_si_unit_ids.tsv", delimiter='\t')

		for property_name in self.sorting.get_property_keys():
			if property_name.startswith('lussac_'):
				unit_ids = new_unit_ids['cluster_id'][np.argmax(new_unit_ids['si_unit_id'].values == self.sorting.unit_ids[:, None], axis=1)].values
				self.write_tsv_file(output_folder / f"{property_name}.tsv", property_name, unit_ids, self.sorting.get_property(property_name))

		if 'estimate_contamination' in params:
			estimated_cont = self._estimate_units_contamination(params['estimate_contamination'])
			unit_ids = np.array(list(estimated_cont.keys()))
			unit_ids = new_unit_ids['si_unit_id'][np.argmax(new_unit_ids['cluster_id'].values == unit_ids[:, None], axis=1)].values
			self.write_tsv_file(output_folder / "lussac_contamination.tsv", "lussac_cont (%)", unit_ids, 100 * np.array(list(estimated_cont.values())))

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

	def _estimate_units_contamination(self, refractory_periods: dict[str, tuple[float, float]]) -> dict[Any, float]:
		"""
		Returns the estimated contamination for each unit (refractory period can vary with the category).

		@param refractory_periods: dict[str, tuple[float, float]]
			The refractory periods for each category, given as [censored_period, refractory_period] (in ms).
		@return estimated_contamination: dict[Any, float]
			The estimated contamination for each unit.
		"""

		estimated_contamination = {}

		for category, refractory_period in refractory_periods.items():
			unit_ids = LussacPipeline.get_unit_ids_for_category(category, self.sorting)
			wvf_extractor = si.WaveformExtractor(self.recording, self.sorting.select_units(unit_ids), allow_unfiltered=True)
			cont, _ = sqm.compute_refrac_period_violations(wvf_extractor, refractory_period[1], refractory_period[0])
			estimated_contamination.update(cont)

		return estimated_contamination

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

		path.parent.mkdir(parents=True, exist_ok=True)

		with open(path, 'w+') as tsv_file:
			tsv_file.write(f"cluster_id\t{name}")
			for key, value in zip(keys, values):
				tsv_file.write(f"\n{key}\t{value}")

			tsv_file.close()
