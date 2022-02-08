import abc
import numpy as np
import spikeextractors


class SpikeSorter(metaclass=abc.ABCMeta):

	def __init__(self, params: dict, output_folder: str):
		"""

		"""

		self.recording = spikeextractors.BinDatRecordingExtractor(params['file'], params['sampling_rate'], params['n_channels'], params['dtype'])
		data_cmr = self.recording.get_traces() - np.median(self.recording.get_traces(), axis=0)[None, :].astype(params['dtype']) # Common median reference.
		self.recording._timeseries = data_cmr
		self.recording = self.recording.load_probe_file(params['prb'])

		self.data_params = params
		self.output_folder = output_folder


	@abc.abstractmethod
	def launch(self, name: str, params: dict={}):
		"""

		"""
