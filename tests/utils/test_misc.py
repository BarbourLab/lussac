import numpy as np
from lussac.utils import estimate_contamination, gaussian_histogram, Utils


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
	spike_train = np.cumsum(t_r*1e-3*sf + np.random.exponential(sf / firing_rate - t_r*1e-3*sf, size=int(2*firing_rate / sf * Utils.t_max)))
	end = np.searchsorted(spike_train, Utils.t_max, side="left")
	spike_train = spike_train[:end].round().astype(np.int64)
	n_contaminated_spikes = int(round(C / (1 - C) * len(spike_train)))

	contaminations = np.empty(10, dtype=np.float32)
	for i in range(len(contaminations)):
		contaminated_spikes = np.random.randint(low=0, high=Utils.t_max, size=n_contaminated_spikes)
		contaminated_spike_train = np.sort(np.concatenate((spike_train, contaminated_spikes)))
		contaminations[i] = estimate_contamination(contaminated_spike_train, (0, 0.8*t_r))

	assert np.abs(np.mean(contaminations) - C) < 0.01

	# Todo: remove duplicated spikes and test with t_c != 0
