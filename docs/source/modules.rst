Lussac modules
==============

Here, we describe each of Lussac modules, what they do, how they do it, and what parameters are available.


The :code:`units_categorization` module
---------------------------------------

This category will label units as belonging to a certain category if they meet some criteria. If a unit already belongs to a category, it will not be re-categorized. So the order matters!
tion
This module takes as a key the name of the category, and as a value a dictionary containing the criteria. Each criterion return a value for each unit, and a minimum and/or maximum can be set.

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
