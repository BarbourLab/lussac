import math
import numpy as np


def _score(x: float, start: float=0.0, end: float=1.0, half_point: float=0.5, penalty: float=0.001):
	"""
	Score function made based on Gompertz function.
	Ressembles a sigmoid, but with more parameters to give what we want.

	@param x (float):
		Calculates score at this point.
	@param start (float):
		Value of score at -inf.
	@param end (float):
		Value of score at +inf.
	@param half_point (float):
		Score is (start+end)/2 at this value.
	@param penalty (float):
		Score at 0 is start +- penalty.
		The smaller the penalty, the faster the change from start to end.

	@return float:
		Score(x)
	"""

	if half_point <= 0:
		print("Error in postprocessing.scores._score():")
		print("half_point should be positive, but is actually {0}".format(half_point))
		return 0

	a = math.log(math.log(penalty)/math.log(0.5)) / half_point
	return start + (end-start) * math.exp(-math.log(2) * math.exp(-a * (x - half_point)))




def get_firing_rate_score(firing_rate: np.ndarray, half_score: float=0.35, penalty: float=0.0005):
	"""
	Returns a score about how good the firing rate is (i.e. how consistant it is).
	1 = very good, 0 = very bad.
	The score is based on the variability standard_deviation/mean.

	@param firing_rate (np.ndarray):
		Array containing the firing rate (see postprocessing.utils.get_firing_rate).
	@param half_score (float):
		If the variability = half_score, then returns 0.5. (see _score).
	@param penalty (float):
		The smaller it is, the faster it will go from 1 to 0. If too big, variability of 0 won't return 1. (see _score).

	@return float (range 0 to 1)
		Float (1 = good firing rate ; 0 = bad firing rate).
	"""

	variability = np.std(firing_rate) / np.mean(firing_rate)

	return _score(variability, start=1.0, end=0.0, half_point=half_score, penalty=penalty)


def get_contamination_score(contamination: float, half_score: float=0.15, penalty: float=0.01):
	"""
	Returns a score about how bad the unit is contaminated.
	1 = not contaminated, 0 = very contaminated.
	The score is based on the estimated contamination (see utils.estimate_contamination).

	@param contamination (float):
		Contamination ratio (between 0 and 1).
	@param half_score (float):
		If the contamination ratio = half_score, then return 0.5. (see _score).
	@param penalty (float):
		The smaller it is, the faster it will go from 1 to 0. (see _score).

	@return float (range 0 to 1)
		Float (1 = not contaminated, 0 = very contaminated unit).
	"""

	return _score(contamination, start=1.0, end=0.0, half_point=half_score, penalty=penalty)

