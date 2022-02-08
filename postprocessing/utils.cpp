#include <iostream>
#include <string>
#include <math.h>
#include <cstdint>


extern "C" {

/**
 * Computes the ISI histogram of a spike train.
 * Takes ~3ms for 200k spikes.
 *
 * @param spike_train Unit's spikes' time (in sampling time)
 * @param n_spikes Number of spikes in the unit
 * @param bin_size Size of bin for histogram (in sampling time)
 * @param max_time Time limit for histogram (in sampling time)
 * @return unsigned int[] containing the ISI.
 */
std::uint32_t* compute_ISI(std::uint64_t* spike_train, std::uint32_t n_spikes, std::uint32_t bin_size, std::uint32_t max_time) {
	max_time -= max_time % bin_size;
	std::uint32_t n_bins = max_time / bin_size;
	std::uint32_t* ISI = new std::uint32_t[n_bins];

	for(std::uint32_t i = 0; i < n_bins; i++) {
		ISI[i] = 0;
	}

	for(std::uint32_t i = 0; i < n_spikes-1; i++) {
		std::uint32_t diff = spike_train[i+1] - spike_train[i];

		if(diff >= max_time)
			continue;

		std::uint32_t bin = diff / bin_size;
		ISI[bin]++;
	}

	return ISI;
}


/**
 * Computes the auto correlogram of a spike train.
 * Takes ~10ms for 200k spikes.
 *
 * @param spike_train Unit's spikes' time (in sampling time)
 * @param n_spikes Number of spikes in the unit
 * @param bin_size Size of bin for histogram (in sampling time)
 * @param max_time Time limit for histogram (in sampling time)
 * @return unsigned int[] containing the auto correlogram.
 */
std::uint32_t* compute_autocorr(std::uint64_t* spike_train, std::uint32_t n_spikes, std::uint32_t bin_size, std::uint32_t max_time) {
	max_time -= max_time % bin_size;
	std::uint32_t n_bins = 2 * (max_time / bin_size);
	std::uint32_t* auto_corr = new std::uint32_t[n_bins];

	for(std::uint32_t i = 0; i < n_bins; i++) {
		auto_corr[i] = 0;
	}

	for(std::uint32_t i = 0; i < n_spikes; i++) {
		for(std::uint32_t j = i+1; j < n_spikes; j++) {
			std::uint32_t diff = spike_train[j] - spike_train[i];

			if(diff >= max_time)
				break;

			std::uint32_t bin = diff / bin_size;
			auto_corr[n_bins/2 - bin - 1]++;
			auto_corr[n_bins/2 + bin]++;
		}
	}

	return auto_corr;
}


/**
 * Computes the cross correlogram of two train spikes.
 * Takes ~2ms for 200k - 3k spikes.
 *
 * @param spike_train1 First unit's spikes' time (in sampling time)
 * @param spike_train2 Second unit's spikes' time (in sampling time)
 * @param n_spikes1 Number of spikes in the first unit
 * @param n_spikes2 Number of spikes in the second unit
 * @param bin_size Size of bin for histogram (in sampling time)
 * @param max_time Time limit for histogram (in sampling time)
 * @return unsigned int[] containing the cross correlogram.
 */
std::uint32_t* compute_crosscorr(std::uint64_t* spike_train1, std::uint64_t* spike_train2, std::uint32_t n_spikes1, std::uint32_t n_spikes2, std::int32_t bin_size, std::int32_t max_time) {
	max_time -= max_time % bin_size;
	std::uint16_t n_bins = 2 * (max_time / bin_size);
	std::uint32_t* cross_corr = new std::uint32_t[n_bins];

	for(std::uint32_t i = 0; i < n_bins; i++) {
		cross_corr[i] = 0;
	}

	std::uint32_t start_j = 0;
	for(std::uint32_t i = 0; i < n_spikes1; i++) {
		for(std::uint32_t j = start_j; j < n_spikes2; j++) {
			std::int32_t diff = (std::int64_t)(spike_train1[i]) - (std::int64_t) spike_train2[j];

			if(diff >= max_time) {
				start_j++;
				continue;
			} else if(diff <= -max_time) {
				break;
			}

			std::int32_t bin = diff / bin_size - (diff >= 0 ? 0 : 1);
			cross_corr[n_bins/2 + bin]++;
		}
	}

	return cross_corr;
}


/**
 * Computes the firing rate correlogram of a spike train.
 * 
 * @param spike_train Unit's spikes' timings (in sampling time)
 * @param n_spikes Number of spikes in the unit
 * @param bin_size Size of a bin in the histogram (in sampling time)
 * @param n_bins Total number of bins in the histogram.
 * @return unsigned int[] containing the firing rate correlogram.
 */
std::uint32_t* compute_firing_rate(std::uint64_t* spike_train, std::uint32_t n_spikes, std::uint32_t bin_size, std::uint32_t n_bins) {
	std::uint32_t* firing_rate = new std::uint32_t[n_bins];

	for(std::uint32_t i = 0; i < n_bins; i++) {
		firing_rate[i] = 0;
	}

	for(std::uint32_t i = 0; i < n_spikes; i++) {
		std::uint32_t bin = spike_train[i] / bin_size;
		if(bin >= n_bins) {
			std::cout << "ERROR: in cpp.compute_firing_rate()" << std::endl;
			std::cout << "\tSpike timing exceeds t_max." << std::endl;
			break;
		}
		firing_rate[bin]++;

	}

	return firing_rate;
}


/**
 * Computes the number of pairs of spikes that are in the refractory period.
 * Takes ~1ms for 200k spikes.
 *
 * @param spike_train Unit's spikes' time (in sampling time)
 * @param n_spikes Number of spikes in the unit
 * @param lower_bound Refractory period's lower bound (in sampling time)
 * @param upper_bound Refractory period's upper bound (in sampling time)
 * @return unsigned int = number of pairs of spikes in refractory period.
 */
std::uint32_t compute_spikes_refractory_period(std::uint64_t* spike_train, std::uint32_t n_spikes, std::uint32_t lower_bound, std::uint32_t upper_bound) {
	std::uint32_t nb_pairs = 0;

	for(std::uint32_t i = 0; i < n_spikes; i++) {
		for(std::uint32_t j = i+1; j < n_spikes; j++) {
			std::uint32_t diff = spike_train[j] - spike_train[i];

			if(diff <= lower_bound)
				continue;
			if(diff >= upper_bound)
				break;

			nb_pairs++;
		}
	}

	return nb_pairs;
}


/**
 * Computes the number of coincident spikes between two spike trains.
 *
 * @param spike_train1 First unit's spikes' time (in sampling time)
 * @param spike_train2 Second unit's spikes' time (in sampling time)
 * @param n_spikes1 Number of spikes in the first unit
 * @param n_spikes2 Number of spikes in the second unit
 * @param max_time Window to consider spikes to be coincident (in sampling time). 0 = need to have exact timing
 * @return unsigned int = number of coincident spikes between the two spike trains.
 */
std::uint32_t compute_nb_coincident_spikes(std::uint64_t* spike_train1, unsigned long* spike_train2, std::uint32_t n_spikes1, std::uint32_t n_spikes2, std::int32_t max_time) {
	std::uint32_t nb_pairs = 0;

	std::uint32_t start_j = 0;
	for(std::uint32_t i = 0; i < n_spikes1; i++) {
		for(std::uint32_t j = start_j; j < n_spikes2; j++) {
			std::int32_t diff = (std::int64_t)(spike_train1[i]) - (std::int64_t)spike_train2[j];

			if(diff > max_time) {
				start_j++;
				continue;
			} else if(diff < -max_time) {
				break;
			}

			nb_pairs++;
		}
	}

	return nb_pairs;
}


/**
 * Computes the numbers of pairs where a spike from spike_train1 is in the refractory period of a spike in spike_train2.
 * 
 * @param spike_train1 First unit's spikes' time (in sampling time)
 * @param spike_train2 Second unit's spikes' time (in sampling time)
 * @param n_spikes1 Number of spikes in the first unit
 * @param n_spikes2 Number of spikes in the second unit
 * @param lower_bound Refractory period's lower bound (in sampling time)
 * @param upper_bound Refractory period's upper bound (in sampling time)
 * @return unsigned int = number of pairs of spikes in the refractory period.
 */
std::uint32_t compute_cross_refractory_period(std::uint64_t* spike_train1, std::uint64_t* spike_train2, std::uint32_t n_spikes1, std::uint32_t n_spikes2, std::uint32_t lower_bound, std::uint32_t upper_bound) {
	std::uint32_t nb_pairs = 0;

	std::uint32_t start_j = 0;
	for(std::uint32_t i = 0; i < n_spikes1; i++) {
		for(std::uint32_t j = start_j; j < n_spikes2; j++) {
			std::int32_t diff = (std::int64_t)(spike_train1[i]) - (std::int64_t) spike_train2[j];

			if(diff > (std::int32_t) upper_bound) {
				start_j++;
				continue;
			} else if(diff < -(std::int32_t) upper_bound) {
				break;
			}

			if(diff >= (std::int32_t) lower_bound || diff <= -(std::int32_t) lower_bound) {
				nb_pairs++;
			}
		}
	}

	return nb_pairs;
}



}
