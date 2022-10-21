import math
import numpy as np
import numba
from spikeinterface.qualitymetrics.misc_metrics import _compute_nb_violations_numba
from .variables import Utils


def gaussian_histogram(events: np.ndarray, t_axis: np.ndarray, sigma: float, truncate: float = 5., margin_reflect: bool = False) -> np.ndarray:
	"""
	Computes a gaussian histogram for the given events.
	For each point in time, take all the nearby events and compute the sum of their gaussian kernel.

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


def estimate_contamination(spike_train: np.ndarray, refractory_period: tuple[float, float]) -> float:
	"""
	Estimates the contamination of a spike train by looking at the number of refractory period violations.
	The spike train is assumed to have spikes coming from a neuron, and noisy spikes that are random and
	uncorrelated to the neuron. Under this assumption, we can estimate the contamination (i.e. the
	fraction of noisy spikes to the total number of spikes).

	@param spike_train: np.ndarray
		The unit's spike train.
	@param refractory_period: tuple[float, float]
		The censored and refractory period (t_c, t_r) used (in ms).
	@return estimated_contamination: float
		The estimated contamination between 0 and 1.
	"""

	t_c = refractory_period[0] * 1e-3 * Utils.sampling_frequency
	t_r = refractory_period[1] * 1e-3 * Utils.sampling_frequency
	n_v = _compute_nb_violations_numba(spike_train.astype(np.int64), t_r)

	N = len(spike_train)
	T = Utils.t_max
	D = 1 - n_v * (T - 2*N*t_c) / (N**2 * (t_r - t_c))
	contamination = 1.0 if D < 0 else 1 - math.sqrt(D)

	return contamination


def estimate_cross_contamination(spike_train1: np.ndarray, spike_train2: np.ndarray, refractory_period: tuple[float, float]) -> float:
	"""
	TODO

	@param spike_train1: np.ndarray
		The spike train of the first unit.
	@param spike_train2: np.ndarray
		The spike train of the second unit.
	@param refractory_period: tuple[float, float]
		The censored and refractory period (t_c, t_r) used (in ms).
	@return (mean_cross_cont, std_cross_cont): tuple[float, float]
		TODO
	"""
	spike_train1 = spike_train1.astype(np.int64)
	spike_train2 = spike_train2.astype(np.int64)

	t_c = refractory_period[0] * 1e-3 * Utils.sampling_frequency
	t_r = refractory_period[1] * 1e-3 * Utils.sampling_frequency
	n_violations = compute_nb_coincidence(spike_train1, spike_train2, t_r) - compute_nb_coincidence(spike_train1, spike_train2, t_c)

	C1 = estimate_contamination(spike_train1, refractory_period)
	C2 = estimate_contamination(spike_train2, refractory_period)

	# Estimate the expected number of violations under the assumption of the 2 spike trains coming from the same neuron,
	# and then under the assumption that they come from different neurons.
	expected_same = 2 * len(spike_train1) * len(spike_train2) * (t_r - t_c) * (C1 + C2 - C1*C2) / Utils.t_max
	expected_diff = 2 * len(spike_train1) * len(spike_train2) * (t_r - t_c) / Utils.t_max

	# TODO: statistical test.

	return (n_violations - expected_same) / (expected_diff - expected_same)


@numba.jit((numba.int64[:], numba.int64[:], numba.float32), nopython=True, nogil=True, cache=True)
def compute_nb_coincidence(spike_train1, spike_train2, max_time):
	"""
	Computes the number of coincident spikes between two spike trains.
	Spike timings are integers, to their real timing follows a uniform distribution between t - dt/2 and t + dt/2.
	Under the assumption that the uniform distributions from two spikes are independent, we can compute the probability
	of those two spikes being closer than the coincidence window:
	f(x) = 1/2 (x+1)² if -1 <= x <= 0
	f(x) = 1/2 (1-x²) + x if 0 <= x <= 1

	@param spike_train1: array[int64] (n_spikes1)
		The spike train of the first unit.
	@param spike_train2: array[int64] (n_spikes2)
		The spike train of the second unit.
	@param max_time: float32
		The maximum time to consider (in number samples).
	@return nb_coincidence: float
		The number of coincident spikes.
	"""

	border_high = math.ceil(max_time)
	border_low = math.floor(max_time)
	p_high = .5 * (max_time - border_high + 1) ** 2
	p_low  = .5 * (1 - (max_time - border_low)**2) + (max_time - border_low)
	nb_coincident = 0
	nb_coincident_low = 0
	nb_coincident_high = 0

	start_j = 0
	for i in range(len(spike_train1)):
		for j in range(start_j, len(spike_train2)):
			diff = spike_train1[i] - spike_train2[j]

			if diff > border_high:
				start_j += 1
				continue
			if diff < -border_high:
				break
			if abs(diff) == border_high:
				nb_coincident_high += 1
			elif abs(diff) == border_low:
				nb_coincident_low += 1
			else:
				nb_coincident += 1

	return nb_coincident + p_high * nb_coincident_high + p_low * nb_coincident_low
