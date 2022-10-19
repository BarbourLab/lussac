import numpy as np
import numba


def gaussian_histogram(events: np.ndarray, t_axis: np.ndarray, sigma: float, truncate: float = 5., margin_reflect: bool=False) -> np.ndarray:
	"""
	Computes a gaussian histogram for the given events.

	@param events: np.ndarray
		The events to histogram.
	@param t_axis: np.ndarray
		The time axis of the histogram.
	@param sigma: float
		The standard deviation of the gaussian kernel (same unit as 't_axis').
	@param truncate: float
		Truncate the gaussian kernel at 'truncate' standard deviation.
	@param margin_reflect: bool
		If true, will reflect the events at the margins.
	@return histogram: np.ndarray
		The histogram of the events.
	"""

	events = np.sort(events).astype(np.float32)
	t_axis = t_axis.astype(np.float32)

	if margin_reflect:
		if np.min(events) >= t_axis[0]:
			events_low = 2*t_axis[0] - events[:np.searchsorted(events, t_axis[0] + truncate * sigma, side="right")][::-1]
		else:
			events_low = np.array([], dtype=np.float32)

		if np.max(events) <= t_axis[-1]:
			events_high = 2*t_axis[-1] - events[np.searchsorted(events, t_axis[-1] - truncate * sigma, side="left"):][::-1]
		else:
			events_high = np.array([], dtype=np.float32)

		events = np.concatenate((events_low, events, events_high))

	return _gaussian_kernel(events, t_axis, sigma, truncate)


@numba.jit((numba.float32[:], numba.float32[:], numba.float32, numba.float32), nopython=True, nogil=True, cache=True, parallel=True)
def _gaussian_kernel(events, t_axis, sigma, truncate):
	"""
	Numba function for gaussian_histogram.

	@param events: array[float32] (n_events)
		The ordered events timings.
	@param t_axis: array[float32] (n_timepoints)
		The time axis of the histogram.
	@param sigma: float32
		The standard deviation of the gaussian kernel (same unit as 't_axis').
	@param truncate: float32
		Truncate the gaussian kernel at 'truncate' standard deviation.
	@return histogram: array[float32] (n_timepoints)
		The gaussian histogram of the events.
	"""

	histogram = np.zeros(t_axis.shape, dtype=np.float32)

	start_j = 0
	for i in numba.prange(len(t_axis)):
		for e in events[start_j:]:
			if e < t_axis[i] - truncate * sigma:
				start_j += 1
				continue
			if e > t_axis[i] + truncate * sigma:
				break

			histogram[i] += np.exp(-0.5 * ((e - t_axis[i]) / sigma) ** 2)

	return histogram / (sigma * np.sqrt(2*np.pi))
