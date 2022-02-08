import os
import glob
import spiketoolkit

from .spike_sorter import SpikeSorter


class Kilosort(SpikeSorter):
	"""
	Abstract class!

	Do not use. use KilosortX where X is the version.
	"""

	def __init__(self, params: dict, output_folder: str):
		super().__init__(params, output_folder)


	def launch(self, name: str, params: dict={}):
		"""

		"""

		if os.path.exists(self.output_folder + "/output"):
			assert False, "Error: output folder '{0}/output' already exists, and would have been overwritten!".format(self.output_folder)

		recording = self.recording
		if 'freq_max' in params:
			recording = spiketoolkit.preprocessing.bandpass_filter(recording, freq_min=0, freq_max=params['freq_max'], dtype=self.data_params['dtype'])

		sorter = self.sorter(recording=recording, output_folder=self.output_folder + "/output")

		sorter.params.update(params)
		sorter.run()
		
		os.remove("{0}/output/recording.dat".format(self.output_folder))
		os.remove("{0}/output/temp_wh.dat".format(self.output_folder))
		for file in glob.glob("{0}/output/*.mat".format(self.output_folder)):
			os.remove(file)
		for file in glob.glob("{0}/output/*.m".format(self.output_folder)):
			os.remove(file)

		os.rename("{0}/output".format(self.output_folder), "{0}/{1}".format(self.output_folder, name))
