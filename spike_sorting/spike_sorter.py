import abc
import numpy as np
import spikeextractors
import spiketoolkit


class SpikeSorter(metaclass=abc.ABCMeta):

	def __init__(self, params: dict, output_folder: str):
		"""

		"""

		self.recording = spikeextractors.BinDatRecordingExtractor(params['file'], params['sampling_rate'], params['n_channels'], params['dtype'])
		self.recording = spiketoolkit.preprocessing.common_reference(self.recording, reference="median")  # Common Median Reference.
		self.recording = self.recording.load_probe_file(params['prb'])

		self.data_params = params
		self.output_folder = output_folder


	@abc.abstractmethod
	def launch(self, name: str, params: dict={}):
		"""

		"""
