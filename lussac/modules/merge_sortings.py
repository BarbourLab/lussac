import numpy as np
import numba
import spikeinterface.core as si
from lussac.core.module import MultiSortingsModule


class MergeSortings(MultiSortingsModule):
	"""
	Merges the sortings into a single one.
	"""

	def run(self, params: dict) -> dict[str, si.BaseSorting]:
		similarity_matrices = self._compute_similarity_matrices(params)

		return self.sortings

	def _compute_similarity_matrices(self, max_time: int) -> dict[str, dict[str, np.ndarray]]:
		"""
		Computes the similarity matrix between all sortings.

		@param max_time: int
			The maximum time difference between spikes to be considered similar.
			Two spikes spaced by exactly max_time are considered coincident.
		@return similarity_matrices: dict[str, dict[str, np.ndarray]]
			The similarity matrices [sorting1, sorting2, similarity_matrix].
		"""

		similarity_matrices = {}
		spike_vectors = {name: sorting.to_spike_vector() for name, sorting in self.sortings.items()}
		n_spikes = {name: np.array([len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.unit_ids]) for name, sorting in self.sortings.items()}

		for name1, sorting1 in self.sortings.items():
			similarity_matrices[name1] = {}
			for name2, sorting2 in self.sortings.items():
				if name1 == name2:
					continue

				coincidence_matrix = self._compute_coincidence_matrix(spike_vectors[name1]['sample_ind'], spike_vectors[name1]['unit_ind'],
																	  spike_vectors[name2]['sample_ind'], spike_vectors[name2]['unit_ind'], max_time)

				similarity_matrix = coincidence_matrix / np.minimum(n_spikes[name1][:, None], n_spikes[name2])
				expected_matrix = n_spikes[name1][:, None] * n_spikes[name2] * (2*max_time+1) / self.recording.get_num_frames()
				similarity_matrices[name1][name2] = (similarity_matrix - expected_matrix) / (1 - expected_matrix)

		return similarity_matrices

	@staticmethod
	@numba.jit((numba.int64[:], numba.int64[:], numba.int64[:], numba.int64[:], numba.int32),
			   nopython=True, cache=True, parallel=True)
	def _compute_coincidence_matrix(spike_times1, spike_labels1, spike_times2, spike_labels2, max_time):
		"""
		Computes the number of coincident spikes between all units in two sortings

		@param spike_times1: array[int64] (n_spikes1)
		@param spike_labels1: array[int64] (n_spikes1)
		@param spike_times2: array[int64] (n_spikes2)
		@param spike_labels2: array[int64] (n_spikes2)
		@param max_time: int32
			The maximum time difference between two spikes to be considered coincident.
			Two spikes spaced by exactly max_time are considered coincident.
		@return coincidence_matrix: array[int64] (n_units1, n_units2)
		"""

		n_units1 = np.max(spike_labels1) + 1
		n_units2 = np.max(spike_labels2) + 1
		coincidence_matrix = np.zeros((n_units1, n_units2), dtype=np.int64)

		start_j = 0
		for i in range(len(spike_times1)):
			for j in range(start_j, len(spike_times2)):
				diff = spike_times1[i] - spike_times2[j]

				if diff > max_time:
					start_j += 1
					continue
				if diff < -max_time:
					break

				coincidence_matrix[spike_labels1[i], spike_labels2[j]] += 1

		return coincidence_matrix
