import numpy as np
import numpy.typing as npt
import lussac.utils as utils
from lussac.utils.misc import _get_border_probabilities
import spikeinterface.core as si


def test_gaussian_histogram() -> None:
	dt = 2e-4
	histogram1 = utils.gaussian_histogram(np.array([3.0, 10.0, 20.0]), np.arange(0, 20 + dt, dt), 0.5)
	histogram2 = utils.gaussian_histogram(np.array([3.0, 10.0, 19.8]), np.arange(0, 20 + dt, dt), 1.0, margin_reflect=True)
	histogram3 = utils.gaussian_histogram(np.array([-40.0, 3.0, 10.0, 20.0, 50.0]), np.arange(0, 20 + dt, dt), 0.5, margin_reflect=True)

	assert np.abs(np.sum(histogram1) * dt - 2.5) < 1e-3
	assert np.abs(np.sum(histogram2) * dt - 3.0) < 1e-3
	assert np.abs(np.sum(histogram3) * dt - 2.5) < 1e-3


def test_get_border_probabilities() -> None:
	assert np.allclose(_get_border_probabilities(0.02), (0, 1, 0.0396, 0.0002))
	assert np.allclose(_get_border_probabilities(0.4), (0, 1, 0.64, 0.08))
	assert np.allclose(_get_border_probabilities(1), (1, 1, 0.5, 0.5))
	assert np.allclose(_get_border_probabilities(3.5), (3, 4, 0.875, 0.125))
	assert np.allclose(_get_border_probabilities(4.73), (4, 5, 0.96355, 0.26645))


def test_estimate_contamination() -> None:
	sf = utils.Utils.sampling_frequency
	firing_rate = 50  # Hz
	t_r = 2.0
	C = 0.1

	spike_train = generate_spike_train(firing_rate, t_r)

	contaminations = np.empty(50, dtype=np.float32)
	for i in range(len(contaminations)):
		contaminated_spikes = generate_contamination(len(spike_train), C)
		contaminated_spike_train = np.sort(np.concatenate((spike_train, contaminated_spikes)))
		contaminations[i] = utils.estimate_contamination(contaminated_spike_train, (0, 0.9*t_r))

	assert np.abs(np.mean(contaminations) - C) < 0.01

	# Todo: remove duplicated spikes and test with t_c != 0


def test_estimate_cross_contamination() -> None:
	sf = utils.Utils.sampling_frequency
	firing_rates = (20, 10)  # Hz
	C = (0.04, 0.06)
	t_r = 2.0
	cross_contamination = 0.6

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

	assert np.abs(np.mean(cross_contaminations) - cross_contamination) < 0.1

	# TODO: remove duplicated spikes and test with t_c != 0


def test_compute_coincidence_matrix() -> None:
	# Test with known result.
	sorting1 = si.NumpySorting.from_dict({0: np.array([18, 163, 622, 1197]), 1: np.array([161, 300, 894])}, sampling_frequency=30000)
	sorting2 = si.NumpySorting.from_dict({0: np.array([120, 298, 303, 628]), 1: np.array([84, 532, 1092])}, sampling_frequency=30000)
	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), 8)

	assert coincidence_matrix[0, 0] == 1
	assert coincidence_matrix[0, 1] == 0
	assert coincidence_matrix[1, 0] == 2

	# Test with random data.
	sf = 30000
	t_max = 3600	# s
	f = 50.			# Hz
	T = t_max * sf
	n_spikes = int(round(t_max * f))
	window = 5		# Number of samples.

	sorting1 = si.NumpySorting.from_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	sorting2 = si.NumpySorting.from_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), window)

	mean = n_spikes**2 * (2*window + 1) / T
	std = np.sqrt(mean)  # p is close to zero so 1-p is one.

	assert np.min(coincidence_matrix) >= mean - 5*std
	assert np.max(coincidence_matrix) <= mean + 5*std
	assert mean - 2*std < np.mean(coincidence_matrix) < mean + 2*std


def test_compute_similarity_matrix() -> None:
	# Test with known result.
	window = 5
	sorting1 = si.NumpySorting.from_dict({0: np.array([18, 163, 622, 1197]), 1: np.array([161, 300, 894])}, sampling_frequency=30000)
	sorting2 = si.NumpySorting.from_dict({0: np.array([155, 304, 628]), 1: np.array([17, 164, 622])}, sampling_frequency=30000)
	n_spikes1 = np.array(list(sorting1.get_total_num_spikes().values()))
	n_spikes2 = np.array(list(sorting2.get_total_num_spikes().values()))

	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), window)
	similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes1, n_spikes2, window)

	assert np.allclose(similarity_matrix, np.array([[0.0, 1.0], [1/3, 1/3]]), atol=1e-4, rtol=1e-4)

	# Test with random data.
	sf = 30000
	t_max = 3600	# s
	f = 50.			# Hz
	T = utils.Utils.t_max
	n_spikes = int(round(t_max * f))
	window = 5		# Number of samples.

	sorting1 = si.NumpySorting.from_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	sorting2 = si.NumpySorting.from_dict({unit_id: np.sort(np.random.randint(low=0, high=T, size=n_spikes)) for unit_id in range(3)}, sf)
	n_spikes1 = np.array(list(sorting1.get_total_num_spikes().values()))
	n_spikes2 = np.array(list(sorting2.get_total_num_spikes().values()))

	coincidence_matrix = utils.compute_coincidence_matrix_from_vector(sorting1.to_spike_vector(), sorting2.to_spike_vector(), window)
	similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes1, n_spikes2)
	corrected_similarity_matrix = utils.compute_similarity_matrix(coincidence_matrix, n_spikes1, n_spikes2, window)

	assert np.abs(np.mean(corrected_similarity_matrix)) < 1e-3
	# TODO: Add tests for uncorrected similarity_matrix


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
