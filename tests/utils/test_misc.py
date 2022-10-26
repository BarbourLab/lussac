import numpy as np
import numpy.typing as npt
from lussac.utils import estimate_contamination, estimate_cross_contamination, gaussian_histogram, Utils


def test_gaussian_histogram() -> None:
	dt = 2e-4
	histogram1 = gaussian_histogram(np.array([3.0, 10.0, 20.0]), np.arange(0, 20 + dt, dt), 0.5)
	histogram2 = gaussian_histogram(np.array([3.0, 10.0, 19.8]), np.arange(0, 20 + dt, dt), 1.0, margin_reflect=True)
	histogram3 = gaussian_histogram(np.array([-40.0, 3.0, 10.0, 20.0, 50.0]), np.arange(0, 20 + dt, dt), 0.5, margin_reflect=True)

	assert np.abs(np.sum(histogram1) * dt - 2.5) < 1e-3
	assert np.abs(np.sum(histogram2) * dt - 3.0) < 1e-3
	assert np.abs(np.sum(histogram3) * dt - 2.5) < 1e-3


def test_estimate_contamination() -> None:
	sf = Utils.sampling_frequency
	firing_rate = 50  # Hz
	t_r = 2.0
	C = 0.1

	spike_train = generate_spike_train(firing_rate, t_r)

	contaminations = np.empty(50, dtype=np.float32)
	for i in range(len(contaminations)):
		contaminated_spikes = generate_contamination(len(spike_train), C)
		contaminated_spike_train = np.sort(np.concatenate((spike_train, contaminated_spikes)))
		contaminations[i] = estimate_contamination(contaminated_spike_train, (0, 0.9*t_r))

	assert np.abs(np.mean(contaminations) - C) < 0.01

	# Todo: remove duplicated spikes and test with t_c != 0


def test_estimate_cross_contamination() -> None:
	sf = Utils.sampling_frequency
	firing_rates = (20, 10)  # Hz
	C = (0.04, 0.06)
	t_r = 2.0
	cross_contamination = 0.6

	spike_train1 = generated_contaminated_spike_train(firing_rates[0], t_r, C[0])
	spike_train2 = generated_contaminated_spike_train(firing_rates[1], t_r, C[1])
	n_transfer = int(round((1 - cross_contamination) * len(spike_train2) / (cross_contamination - C[0])))

	cross_contaminations = np.empty(50, dtype=np.float32)
	for i in range(len(cross_contaminations)):
		indices = np.sort(np.random.choice(range(len(spike_train1)), n_transfer, replace=False))
		train2 = np.sort(np.concatenate((spike_train2, spike_train1[indices])))
		train1 = np.delete(spike_train1, indices)

		estimation, p_value = estimate_cross_contamination(train1, train2, (0, 0.9*t_r), limit=0.3)
		assert p_value < 1e-6
		cross_contaminations[i] = estimation

	assert np.abs(np.mean(cross_contaminations) - cross_contamination) < 0.02

	# TODO: remove duplicated spikes and test with t_c != 0


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
	sf = Utils.sampling_frequency
	t_r *= 1e-3 * sf
	beta = sf / firing_rate - t_r

	n_spikes = firing_rate / sf * Utils.t_max
	spike_train = np.cumsum(t_r + np.random.exponential(beta, size=int(1.5*n_spikes)))
	end = np.searchsorted(spike_train, Utils.t_max, side="left")

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
	return np.random.randint(low=0, high=Utils.t_max, size=n_contaminated_spikes)


def generated_contaminated_spike_train(firing_rate: float, t_r: float, C: float) -> npt.NDArray[np.int64]:
	"""
	TODO

	@param firing_rate:
	@param t_r:
	@param C:
	@return:
	"""

	spike_train = generate_spike_train(firing_rate, t_r)
	contaminated_spikes = generate_contamination(len(spike_train), C)
	return np.sort(np.concatenate((spike_train, contaminated_spikes)))
