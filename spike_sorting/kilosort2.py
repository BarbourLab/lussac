import spikesorters

from .kilosort import Kilosort


class Kilosort2(Kilosort):

	def __init__(self, params: dict, output_folder: str):
		super().__init__(params, output_folder)

		self.sorter = spikesorters.Kilosort2Sorter
