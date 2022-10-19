import numpy as np
import lussac.utils.misc as misc


def test_gaussian_histogram() -> None:
	dt = 2e-4
	histogram1 = misc.gaussian_histogram(np.array([3.0, 10.0, 20.0]), np.arange(0, 20 + dt, dt), 0.5)
	histogram2 = misc.gaussian_histogram(np.array([3.0, 10.0, 19.8]), np.arange(0, 20 + dt, dt), 1.0, margin_reflect=True)

	assert np.abs(np.sum(histogram1) * dt - 2.5) < 1e-3
	assert np.abs(np.sum(histogram2) * dt - 3.0) < 1e-3
