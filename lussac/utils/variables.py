import pathlib
from typing import ClassVar


class Utils:
	"""
	Class containing static variables that are useful for the whole package.

	Variables:
		sampling_frequency	The sampling frequency of the recording (in Hz).
		t_max				The number of time samples in the recording.
		plotly_js_file		The path to the plotly.js file.

	"""

	sampling_frequency:	ClassVar[int]
	t_max:				ClassVar[int]
	plotly_js_file:		ClassVar[pathlib.Path]
