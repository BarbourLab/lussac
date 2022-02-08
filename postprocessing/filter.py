from multiprocessing import Pool
import numpy as np
import scipy.signal



def get_filter_params(order: int=3, min_f: float=100.0, max_f: float=6000.0, sampling_rate: float=3e4, btype: str="bandpass"):
	"""
	Returns the filter parameters as a tuple (b, a).

	@param order (int):
		The higher the order, the more strict the limits will be.
	@param min_f (float):
		Minimum frequency (Hz) under which the frequencies will start being filtered out.
	@param max_f (float):
		Maximum frequency (Hz) over which the frequencies will start being filtered out.
	@param sampling_rate (float):
		Sampling rate of the arrays the filter will be performed on.
	@param btype (str):
		"lowpass" for low-pass filter (will ignore min_f).
		"highpass" for high-pass filter (will ignore max_f).
		"bandpass" for band-pass filter (will use both min_f and max_f).

	@return b,a (np.ndarray):
		The result of scipy.signal.butter().
		This is what you will pass to the others filter functions.
	"""

	nyq = 0.5 * sampling_rate

	if btype == "lowpass":
		return scipy.signal.butter(order, max_f/nyq, btype=btype)
	if btype == "highpass":
		return scipy.signal.butter(order, min_f/nyq, btype=btype)
	if btype == "bandpass":
		return scipy.signal.butter(order, [min_f/nyq, max_f/nyq], btype=btype)

	print("Incorrect btype for filter.get_filter_params: '{0}'".format(btype))
	assert False


def filter_units_waveforms(units_waveforms: list, b: np.ndarray, a: np.ndarray, dtype):
	"""
	Filters all units waveforms one by one.

	@param units_waveforms (list of np.ndarray):
		List of all unit's waveforms. For more information, see filter.filter_waveforms.
	@param b, a (np.ndarray):
		Result of filter.get_filter_params().
	@param dtype:
		The dtype you want the filtered waveforms to be in.

	@return filtered_units_waveforms (list of np.ndarray):
		List of all unit's filtered waveforms. For more information, see filter.filter_waveforms.
	"""

	filtered_waveforms = []
	for unit_waveforms in units_waveforms:
		filtered_waveforms.append(filter(unit_waveforms, b, a, dtype))

	return filtered_waveforms


def filter(waveforms: np.ndarray, b: np.ndarray, a: np.ndarray, dtype):
	"""
	Filters all the waveforms individually and returns the filtered waveforms.
	Filters both way to not induce any shifts.

	@param waveforms (np.ndarray) [..., time]:
		Waveforms to filter. Will only filter on last axis (time).
	@param b, a (np.ndarray):
		Result of filter.get_filter_params().
	@param dtype:
		The dtype you want the filtered waveforms to be in.

	@return filtered_waveforms (np.ndarray[dtype]) [..., time]:
		Filtered waveforms along time axis.
	"""

	filtered_waveforms = scipy.signal.filtfilt(b, a, waveforms, axis=len(waveforms.shape)-1)
	if np.issubdtype(dtype, np.integer):
		filtered_waveforms = filtered_waveforms.round()

	return filtered_waveforms.astype(dtype)




def filter_units_waveforms_parallel(units_waveforms: list, b: np.ndarray, a: np.ndarray, dtype, n_process: int=8):
	"""
	Parallel implementation of filter.filter_units_waveforms().
	"""

	if n_process == 1:
		return filter_units_waveforms(units_waveforms, b, a, dtype)

	filtered_waveforms = []

	with Pool(processes=n_process) as pool:
		res = []

		for unit_waveforms in units_waveforms:
			res.append(pool.apply_async(filter, (unit_waveforms, b, a, dtype)))

		for i in range(len(units_waveforms)):
			filtered_waveforms.append(res[i].get())

	return filtered_waveforms

