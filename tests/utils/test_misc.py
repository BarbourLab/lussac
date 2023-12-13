import math
import numpy as np
import numpy.typing as npt
import scipy.stats
import lussac.utils as utils
from lussac.utils.misc import _get_border_probabilities
import spikeinterface.core as si
from spikeinterface.curation.curation_tools import find_duplicated_spikes


def test_filter_kwargs() -> None:
	assert utils.filter_kwargs({}, test_flatten_dict) == {}
	assert utils.filter_kwargs({'t_r': 2.0, 't_c': 1.0}, generate_spike_train) == {'t_r': 2.0}


def test_flatten_dict() -> None:
	assert utils.flatten_dict({}) == {}

	d = {'a': 1, 'b': 2}
	assert utils.flatten_dict(d) == d

	d = {'a': {'b': 1, 'c': {'d': 2, 'e': {}}}, 'f': 3}
	assert utils.flatten_dict(d, sep=':') == {'a:b': 1, 'a:c:d': 2, 'a:c:e': {}, 'f': 3}


def test_unflatten_dict() -> None:
	assert utils.flatten_dict({}) == {}

	d = {'a': 1, 'b': 2}
	assert utils.flatten_dict(d) == d

	d = {'a:b': 1, 'a:c:d': 2, 'a:c:e': {}, 'f': 3}
	assert utils.unflatten_dict(d) == {'a': {'b': 1, 'c': {'d': 2, 'e': {}}}, 'f': 3}

	d = {'a': {'b': 1, 'c': {}}, 'd': 3}
	assert utils.unflatten_dict(utils.flatten_dict(d)) == d


def test_merge_dict() -> None:
	d1 = {'a': 1, 'b': [1, 2, 3], 'c': {'d': 2, 'e': 3, 'f': {'g': False}}, 1: 2}
	d2 = {'a': 2, 'b': [1, 2, 4], 'c': {'d': 2, 'e': True, 'f': {'g': {'g': 0}}}, 2: 1}

	d3 = utils.merge_dict(d1, d2)
	d4 = utils.merge_dict(d2, d1)

	for key in d1:
		assert d3[key] == d1[key]
	assert d3[2] == 1

	for key in d2:
		assert d4[key] == d2[key]
	assert d4[1] == 2

	# Order of keys should be preserved.
	d3_keys = np.array(list(d3.keys()))
	d4_keys = np.array(list(d4.keys()))
	assert np.all(d3_keys == ('a', 'b', 'c', 1, 2))
	assert np.all(d4_keys == ('a', 'b', 'c', 2, 1))


def test_binom_sf() -> None:
	x, n, p = 3, 30.5, 0.1
	res = 1 - 0.635629357849

	assert math.isclose(utils.binom_sf(x, math.floor(n), p), scipy.stats.binom.sf(x, math.floor(n), p), rel_tol=1e-5, abs_tol=1e-5)
	assert math.isclose(utils.binom_sf(x, n, p), res, rel_tol=1e-5, abs_tol=1e-5)


def test_gaussian_histogram() -> None:
	dt = 2e-4
	histogram1 = utils.gaussian_histogram(np.array([3.0, 10.0, 20.0]), np.arange(0, 20 + dt, dt), 0.5)
	histogram2 = utils.gaussian_histogram(np.array([3.0, 10.0, 19.8]), np.arange(0, 20 + dt, dt), 1.0, margin_reflect=True)
	histogram3 = utils.gaussian_histogram(np.array([-40.0, 3.0, 10.0, 20.0, 50.0]), np.arange(0, 20 + dt, dt), 0.5, margin_reflect=True)

	assert math.isclose(np.sum(histogram1) * dt, 2.5, rel_tol=1e-3, abs_tol=1e-3)
	assert math.isclose(np.sum(histogram2) * dt, 3.0, rel_tol=1e-3, abs_tol=1e-3)
	assert math.isclose(np.sum(histogram3) * dt, 2.5, rel_tol=1e-3, abs_tol=1e-3)

	# Make sure it doesn't crash if empty.
	assert utils.gaussian_histogram(np.array([], dtype=np.float32), np.arange(0, 20 + dt, dt), 0.5).shape == np.arange(0, 20 + dt, dt).shape


def test_get_border_probabilities() -> None:
	assert np.allclose(_get_border_probabilities(0.02), (0, 1, 0.0396, 0.0002))
	assert np.allclose(_get_border_probabilities(0.4), (0, 1, 0.64, 0.08))
	assert np.allclose(_get_border_probabilities(1), (1, 1, 0.5, 0.5))
	assert np.allclose(_get_border_probabilities(3.5), (3, 4, 0.875, 0.125))
	assert np.allclose(_get_border_probabilities(4.73), (4, 5, 0.96355, 0.26645))


def test_estimate_contamination() -> None:
	firing_rate = 50  # Hz
	t_c = 0.5
	t_r = 2.0
	C = 0.1

	# Test without any censored period.
	spike_train = generate_spike_train(firing_rate, t_r)

	contaminations = np.empty(50, dtype=np.float32)
	for i in range(len(contaminations)):
		contaminated_spikes = generate_contamination(len(spike_train), C)
		contaminated_spike_train = np.sort(np.concatenate((spike_train, contaminated_spikes)))
		contaminations[i] = utils.estimate_contamination(contaminated_spike_train, (0, 0.9*t_r))

	assert math.isclose(np.mean(contaminations), C, rel_tol=1e-2, abs_tol=1e-2)

	# Test with a censored period.
	contaminations = np.empty(100, dtype=np.float32)

	for i in range(len(contaminations)):
		spike_train = generate_censored_contaminated_spike_train(firing_rate, (t_c, t_r), C)
		contaminations[i] = utils.estimate_contamination(spike_train, (t_c, 0.9*t_r))

	assert math.isclose(np.mean(contaminations), C, rel_tol=5e-2, abs_tol=5e-2)


def test_estimate_cross_contamination() -> None:
	firing_rates = (20, 10)  # Hz
	C = (0.04, 0.06)
	t_c = 0.5
	t_r = 2.0
	cross_contamination = 0.6

	# Testing without any censored period.
	spike_train1 = generate_contaminated_spike_train(firing_rates[0], t_r, C[0])
	spike_train2 = generate_contaminated_spike_train(firing_rates[1], t_r, C[1])
	n_transfer = int(round((1 - cross_contamination) * len(spike_train2) / (cross_contamination - C[0])))

	cross_contaminations = np.empty(50, dtype=np.float32)
	for i in range(len(cross_contaminations)):
		indices = np.sort(np.random.choice(range(len(spike_train1)), n_transfer, replace=False))
		train2 = np.sort(np.concatenate((spike_train2, spike_train1[indices])))
		train1 = np.delete(spike_train1, indices)

		estimation, p_value = utils.estimate_cross_contamination(train1, train2, (0, 0.9*t_r), limit=0.3)
		assert p_value < 1e-6
		cross_contaminations[i] = estimation

	assert math.isclose(np.mean(cross_contaminations), cross_contamination, abs_tol=0.1, rel_tol=0.1)

	# Testing with t_c != 0
	# TODO


def test_compute_coincidence_matrix() -> None:
	# Test with known result.
	sorting1 = si.NumpySorting.from_unit_dict({0: np.array([18, 163, 622, 1197]), 1: np.array([161, 300, 894])}, sampling_frequency=30000)
	sorting2 = si.NumpySorting.from_unit_dict({0: np.array([120, 298, 303, 628]), 1: np.array([84, 532, 1092])}, sampling_frequency=30000)
	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), 8)

	assert coincidence_matrix[0, 0] == 1
	assert coincidence_matrix[0, 1] == 0
	assert coincidence_matrix[1, 0] == 2

	# Test with random data.
	sf = 30000		# Hz
	t_max = 3600	# s
	f = 50.			# Hz
	T = t_max * sf
	n_spikes = int(round(t_max * f))
	window = 5		# Number of samples.

	sorting1 = si.NumpySorting.from_unit_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	sorting2 = si.NumpySorting.from_unit_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), window)

	mean = n_spikes**2 * (2*window + 1) / T
	std = np.sqrt(mean)  # p is close to zero so 1-p is one.

	assert np.min(coincidence_matrix) >= mean - 5*std
	assert np.max(coincidence_matrix) <= mean + 5*std
	assert mean - 2*std < np.mean(coincidence_matrix) < mean + 2*std

	# Test with cross-shift.
	sorting1 = si.NumpySorting.from_unit_dict({0: np.array([100, 200, 400])}, sampling_frequency=sf)
	sorting2 = si.NumpySorting.from_unit_dict({0: np.array([106, 206, 406])}, sampling_frequency=sf)
	cross_shift = np.array([[-6]])

	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), 2, None)
	coincidence_shifted = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), 2, cross_shift)

	assert coincidence_matrix[0, 0] == 0
	assert coincidence_shifted[0, 0] == 3


def test_compute_similarity_matrix() -> None:
	# Test with known result.
	window = 5
	sorting1 = si.NumpySorting.from_unit_dict({0: np.array([18, 163, 622, 1197]), 1: np.array([161, 300, 894])}, sampling_frequency=30000)
	sorting2 = si.NumpySorting.from_unit_dict({0: np.array([155, 304, 628]), 1: np.array([17, 164, 622])}, sampling_frequency=30000)
	n_spikes1 = np.array(list(sorting1.count_num_spikes_per_unit().values()))
	n_spikes2 = np.array(list(sorting2.count_num_spikes_per_unit().values()))

	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), window)
	similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes1, n_spikes2, window)

	assert np.allclose(similarity_matrix, np.array([[0.0, 1.0], [1/3, 1/3]]), atol=1e-4, rtol=1e-4)

	# Test with random data.
	sf = 30000		# Hz
	t_max = 3600	# s
	f = 50.			# Hz
	T = utils.Utils.t_max
	n_spikes = int(round(t_max * f))
	window = 5		# Number of samples.

	sorting1 = si.NumpySorting.from_unit_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	sorting2 = si.NumpySorting.from_unit_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	n_spikes1 = np.array(list(sorting1.count_num_spikes_per_unit().values()))
	n_spikes2 = np.array(list(sorting2.count_num_spikes_per_unit().values()))

	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), window)
	similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes1, n_spikes2)
	corrected_similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes1, n_spikes2, window)
	uncorrected_similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes1, n_spikes2)

	# Test for corrected similarity matrix.
	assert math.isclose(np.mean(corrected_similarity_matrix), 0.0, rel_tol=1e-3, abs_tol=1e-3)

	# Test for uncorrected similarity matrix.
	n_expected = n_spikes**2 * (2*window + 1) / T
	assert math.isclose(np.mean(uncorrected_similarity_matrix), n_expected / n_spikes, rel_tol=1e-3, abs_tol=1e-3)


def generate_spike_train(firing_rate: float, t_r: float) -> npt.NDArray[np.int64]:
	"""
	Generates a spike train with a refractory period.

	@param firing_rate: float
		The mean firing rate of the neuron (in Hz).
		Will not be the actual firing rate, but is the target.
	@param t_r: float
		The refractory period (in ms).
	@return spike_train: np.ndarray[int64]
		The generated spike train.
	"""
	sf = utils.Utils.sampling_frequency
	t_r *= 1e-3 * sf
	beta = sf / firing_rate - t_r

	n_spikes = firing_rate / sf * utils.Utils.t_max
	spike_train = np.cumsum(t_r + np.random.exponential(beta, size=int(1.5*n_spikes)))
	end = np.searchsorted(spike_train, utils.Utils.t_max, side="left")

	return spike_train[:end].round().astype(np.int64)


def generate_contamination(n_spikes_neuron: int, C: float) -> npt.NDArray[np.int64]:
	"""
	Generates a contamination spike train.

	@param n_spikes_neuron: int
		The number of spikes in the neuron.
	@param C: float
		The contamination rate.
	@return contamination: np.ndarray[int64]
		The generated contamination spike train.
	"""

	n_contaminated_spikes = int(round(C / (1 - C) * n_spikes_neuron))
	return np.random.randint(low=0, high=utils.Utils.t_max, size=n_contaminated_spikes)


def generate_contaminated_spike_train(firing_rate: float, t_r: float, C: float) -> npt.NDArray[np.int64]:
	"""
	Generates a contaminated spike train.

	@param firing_rate: float
		The mean firing rate of the neuron (in Hz).
		The contamination is added on top of that.
	@param t_r: float
		The refractory period of the neuron (in ms).
	@param C: float
		The contamination rate.
	@return: np.ndarray[int64]
		The generated contaminated spike train.
	"""

	spike_train = generate_spike_train(firing_rate, t_r)
	contaminated_spikes = generate_contamination(len(spike_train), C)
	return np.sort(np.concatenate((spike_train, contaminated_spikes)))


def generate_censored_contaminated_spike_train(firing_rate: float, refractory_period: tuple[float, float], C: float) -> npt.NDArray[np.int64]:
	"""
	Generates a censored contaminated spike train.

	@param firing_rate: float
		The mean firing rate of the neuron (in Hz).
		The contamination is added on top of that.
	@param refractory_period: tuple[float, float]
		The (censored_period, refractory_period) of the neuron (in ms).
	@param C: float
		The contamination rate.
	@return: np.ndarray[int64]
		The generated censored contaminated spike train.
	"""

	t_c = int(round(refractory_period[0] * 1e-3 * utils.Utils.sampling_frequency))
	spike_train = generate_contaminated_spike_train(firing_rate, refractory_period[1], C)
	return np.delete(spike_train, find_duplicated_spikes(spike_train, t_c, method="keep_first_iterative"))


def test_compute_cross_shift() -> None:
	spike_times1 = np.arange(2000, 100000, 2000, dtype=np.int64)
	spike_labels1 = np.zeros(len(spike_times1), dtype=np.int64)
	spike_labels1[1::2] = 1

	# Unit 1 is clearly shifted by 2 samples. Unit 0 has only one spike shifted (doesn't cross the threshold). Unit 2 has no correlation.
	spike_times2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2002, 6002, 10001, 14003, 18002, 22002, 24002, 25648, 30002, 31578], dtype=np.int64)
	spike_labels2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 1, 2], dtype=np.int64)

	cross_shifts = utils.compute_cross_shift(spike_times1, spike_labels1, spike_times2, spike_labels2, 30, 1.5)
	assert np.all(cross_shifts == np.array([[0, -2, 0], [0, 0, 0]]))


def test_filter() -> None:
	# Test that everything is valid when passing a 1d-array.
	xaxis = np.arange(0, 0.1, 1/utils.Utils.sampling_frequency)
	N = len(xaxis)
	data = np.cos(2*np.pi*50*xaxis) + 3 * np.sin(2*np.pi*1000*xaxis) + np.cos(2*np.pi*10000*xaxis + 0.1)
	data_filtered = utils.filter(data, [300, 6000], axis=0)

	freq = np.fft.rfftfreq(N, 1/utils.Utils.sampling_frequency)
	data_filtered_fft = np.abs(np.fft.rfft(data_filtered) * 2 / N)

	freq50 = np.argmax(freq >= 50)
	freq1000 = np.argmax(freq >= 1000)
	freq10000 = np.argmax(freq >= 10000)

	assert data_filtered_fft[freq50] < 0.1
	assert data_filtered_fft[freq1000] > 2.9
	assert data_filtered_fft[freq10000] < 0.3

	# Test that everything is valid with multi-dimensional arrays.
	frequencies = np.array([[50, 500], [1000, 10000]])

	data = np.cos(2*np.pi*frequencies[None, :, :] * xaxis[:, None, None])
	data_filtered = utils.filter(data, [300, 6000], axis=0)
	fft0 = np.abs(np.fft.rfft(data_filtered, axis=0) * 2 / N)

	data = np.cos(2*np.pi*frequencies[:, None, :] * xaxis[None, :, None])
	data_filtered = utils.filter(data, [300, 6000], axis=1)
	fft1 = np.abs(np.fft.rfft(data_filtered, axis=1) * 2 / N)

	data = np.cos(2*np.pi*frequencies[:, :, None] * xaxis[None, None, :])
	data_filtered = utils.filter(data, [300, 6000], axis=2)
	fft2 = np.abs(np.fft.rfft(data_filtered, axis=2) * 2 / N)

	assert np.all(fft0[freq50] < 0.1)
	assert np.all(fft1[:, freq50] < 0.1)
	assert np.all(fft2[..., freq50] < 0.1)

	assert fft0[freq1000, 1, 0] > 0.95
	assert fft1[1, freq1000, 0] > 0.95
	assert fft2[1, 0, freq1000] > 0.95

	assert np.all(fft0[freq10000] < 0.3)
	assert np.all(fft1[:, freq10000] < 0.3)
	assert np.all(fft2[..., freq10000] < 0.3)
