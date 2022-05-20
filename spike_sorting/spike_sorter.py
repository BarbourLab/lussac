from abc import ABC, abstractmethod
import spikeinterface.core as si
import spikeinterface.toolkit as st


class LussacSpikeSorter(ABC):
	"""
	Abstract class for all spike sorting algorithms running through Lussac.
	All children must have names to recognize them, and a launch method.

	Attributes:
		recording		A SpikeInterface recording object.
		output_folder	Path to the folder containing the output of all spike sorters.
	"""

	__slots__ = "recording", "output_folder"
	recording: si.BaseRecording
	output_folder: str

	def __init__(self, recording: si.BaseRecording, output_folder: str) -> None:
		"""
		Creates a new LussacSpikeSorter instance.
		Abstract class for all spike sorting algorithms running through Lussac.

		@param recording: BaseRecording
			The recording object.
		@param output_folder: str
			Path to the folder containing the output of all spike sorter ran through Lussac.
		"""

		self.recording = st.preprocessing.common_reference(recording, reference="global", operator="median")
		self.output_folder = output_folder

	@property
	@abstractmethod
	def names(self) -> tuple:
		"""
		Returns a tuple containing all acceptable names to reference
		to this spike sorter in the Lussac parameters.

		@return names: tuple
		"""
		...

	@abstractmethod
	def launch(self, name: str, params: dict) -> None:
		"""
		Launches the spike sorting algorithm and saves the result as a Phy folder.

		@param name: str
			The output's folder name.
		@param params: dict
			The parameters for the spike sorting algorithm.
		"""
		...
