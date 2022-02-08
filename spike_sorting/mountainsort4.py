import os
import shutil
import spikesorters
import spiketoolkit

from .spike_sorter import SpikeSorter


class MountainSort4(SpikeSorter):

	def launch(self, name: str, params: dict={}):
		"""

		"""

		sorting = spikesorters.run_mountainsort4(recording=self.recording, output_folder=self.output_folder + "/output", **params)
		tmp_folder = "{0}/output/tmp".format(self.output_folder)

		os.makedirs(tmp_folder, exist_ok=True)
		sorting.set_tmp_folder(tmp_folder)
		spiketoolkit.postprocessing.export_to_phy(self.recording, sorting, self.output_folder + "/{0}".format(name), compute_pc_features=False, compute_amplitudes=False, max_channels_per_template=self.data_params['n_channels'],
														max_spikes_per_unit=1000, copy_binary=False, ms_before=1.0, ms_after=3.0, dtype=self.data_params['dtype'], recompute_info=True, n_jobs=3, filter_flag=False)

		del sorting
		shutil.rmtree("{0}/output".format(self.output_folder))
