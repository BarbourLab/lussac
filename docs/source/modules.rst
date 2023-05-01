Lussac modules
==============

Here, we describe each of Lussac modules, what they do, how they do it, and what parameters are available.

TODO: mini section to explain :code:`wvf_extraction`.


The :code:`units_categorization` module
---------------------------------------

This module will label units as belonging to a certain category if they meet some criteria. If a unit already belongs to a category, it will not be re-categorized. So the order matters!

This module takes as a key the name of the category, and as a value a dictionary containing the criteria. Each criterion return a value for each unit, and a minimum and/or maximum can be set.

TODO: Explain filter.

- :code:`firing_rate`: returns the mean firing rate of the unit (in Hz).
- :code:`contamination`: returns the estimated contamination of the unit (between 0 and 1; 0 being pure). The :code:`refractory_period = [censored_period, refractory_period]` has to be set (in ms).
- :code:`amplitude`: returns the mean amplitude of the unit's template (in µV if the recording object can be scaled to µV). Optional parameters can be set to the function :code:`spikeinterface.core.get_template_extremum_amplitude`.
- :code:`SNR`: returns the signal-to-noise ratio of the unit. Optional parameters can be set to the function :code:`spikeinterface.qualitymetrics.compute_snrs`
- :code:`amplitude_std`: returns the standard deviation in the spike amplitudes for the unit. Optional parameters can be set to the function :code:`spikeinterface.postprocessing.compute_spike_amplitudes`
- :code:`ISI_portion`: Returns the fraction (between 0 and 1) of inter-spike intervals inside a time range. the :code:`range = [min_t, max_t]` has to be set (in ms).


Example of categorization
^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example for categorizing complex-spikes from more regular spikes (cerebellar cortex example):

.. code-block:: json

	"units_categorization": {
		"all": {  // Categorize all units.
			"CS": {  // Criteria for complex-spikes category.
				"firing_rate": {  // Firing rate < 5 Hz
					"max": 5.0
				},
				"ISI_portion": {  // Few spikes between 10 and 35 ms in the ISI.
					"range": [10.0, 35.0],
					"max": 0.05
				}
			},
			"spikes": {  // Categorize more "regular" spikes
				"firing_rate": {
					"min": 0.4,
					"max": 200.0
				},
				"contamination": {  // Maximum of 30% contamination.
					"refractory_period": [0.3, 1.0],
					"max": 0.3
				},
				"SNR": {
					"peak_sign": "neg",  // Example of an SI parameter.
					"min": 2.5
				}
			}
		}
	}


Clearing units category
^^^^^^^^^^^^^^^^^^^^^^^

It is possible to remove the category label on units, by setting the category name to :code:`"clear"`. For example:

.. code-block:: json

	"units_categorization": {
		"all": {"clear": {}}  // Clear category label from all units.
	}


The :code:`align_units` module
------------------------------

This module will align units by using their template waveform. The algorithm is not that straightforward:

| First, a threshold is set and we look at the first peak that is higher than this threshold (both in positive and negative values). Next, the algorithm checks a few more samples in time for a higher peak (if one is detected, it takes this one).
| The rational behind it is that the "center" of the spike should be when the neuron starts its action potential. For multi-phasic spikes, this is usually the first one. The :code:`threshold` and :code:`check_next` parameters are here to make sure we're not detecting noise.

TODO: Insert image with example of sub-threshold peak and check_next.

This module's parameters are:

- :code:`wvf_extraction`: to construct the templates. The :code:`ms_before` and :code:`ms_after` parameters determine the max shift for alignment.
- :code:`filter`: the band for the bandpass Gaussian filtering of the templates :code:`[min_f, max_f]`. Can be set to :code:`null` for no filtering.
- :code:`threshold`: Threshold multiplicator (between 0 and 1). The real threshold is :code:`max(template) * threshold`.
- :code:`check_next`: Number of samples to check after the first peak (put 0 to not check after the first peak).


Example of units alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

	"align_units": {
		"all": {  // Align all units.
			"wvf_extraction": {
				"ms_before": 1.0,
				"ms_after": 2.0,
				"max_spikes_per_unit": 2000  // Use 2,000 random spikes to construct templates.
			},
			"filter": [300.0, 6000.0],  // Gaussian-filter between 300 and 6000 Hz.
			"threshold": 0.5,  // Threshold at 50% of the maximum.
			"check_next": 5  // Check the next 5 samples.
		}
	}
