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
- :code:`threshold` (optional): Threshold multiplicator (between 0 and 1). The real threshold is :code:`max(template) * threshold`. By default: 0.5
- :code:`check_next` (optional): Number of samples to check after the first peak (put 0 to not check after the first peak). By default: 10


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


The :code:`remove_bad_units` module
-----------------------------------

This module will remove the units that meet at least one of the criteria. The criteria are the same as those described in :code:`units_categorization`.


Example of units removal
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

	"remove_bad_units": {
		"CS": {  // Remove complex-spike units with contamination > 35%
			"contamination": {
				"refractory_period": [1.5, 25.0],
				"max": 0.35
			}
		},
		"spikes": {  // Remove units with firing rate < 1.0 Hz or amplitude std > 80 µV
			"firing_rate": {
				"min": 1.0
			},
			"amplitude_std": {
				"max": 80.0
			}
		}


The :code:`remove_duplicated_spikes` module
-------------------------------------------

This module will remove spikes that are considered duplicates (i.e. too close to one another).

| This is done by setting a :code:`censored_period` window under which there cannot be 2 spikes.
| Be careful! This is different from the :code:`refractory_period`! It's very useful to keep spikes in the refractory period to estimate the contamination. The censored period is designed to remove duplicated spikes.
| Typical values of :code:`censored_period` usually lie between 0.2 and 0.4 ms, whereas the refractory period is almost always greater than 0.9ms.

This module's parameters are:

- :code:`censored_period`: in ms (by default, 0.3).
- :code:`method` (optional): method used to remove duplicates (used by :code:`spikeinterface.curation.find_duplicated_spikes`). By default: :code:`"keep_first_iterative"`


Example of duplicated spikes removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

	"remove_duplicated_spikes": {
		"all": {
			"censored_period": 0.3
		}
	}


The :code:`remove_redundant_units` module
-----------------------------------------

| This module will look for redundant units in analyses (by looking at the rate of coincident spikes between units in individual analyses).
| If redundant units are detected, all but one will be removed (the chosen one depends on the :code:`remove_strategy` used).

This module's parameters are:

- :code:`wvf_extraction`: to construct the templates (required depending on the remove strategy). If not required, just set it to :code:`null`.
- :code:`arguments`: a :code:`dict` containing the parameters to give to :code:`spikeinterface.curation.remove_redundant_units`.


Example of redundant units removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

	"remove_redundant_units": {
		"all": {
			"wvf_extraction": {
				"ms_before": 1.0,
				"ms_after": 1.5,
				"max_spikes_per_unit": 500
			},
			"arguments": {
				"align": true,  // Can be set to 'false' if you already used the 'align_units' module.
				"delta_time": 0.3,  // Window (in ms) to consider coincident spikes.
				"duplicate_threshold": 0.7,  // If coincidence >= 70%, consider the units redundant.
				"remove_strategy": "highest_amplitude",  // Keep the unit with the highest amplitude.
			}
		}
	}


The :code:`merge_units` module
------------------------------

WIP


The :code:`merge_sortings` module
---------------------------------

WIP


The :code:`export_to_phy` module
--------------------------------

This module will export all sortings in their current state to the :code:`phy` format (if :code:`merge_sortings` was called before, will only export the merged sorting).

This module's parameters are:

- :code:`path`: path to the folder where to export the sorting(s). If multiple sortings exists, a subfolder will be created for each of them.
- :code:`wvf_extraction`: to construct the templates.
- :code:`export_params`: a :code:`dict` containing the parameters to give to :code:`spikeinterface.exporters.export_to_phy`.
- :code:`estimate_contamination` (optional): a :code:`dict` containing the refractory period for each category. If given, will output the estimated contamination of the units.


Example of export to phy
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

	"export_to_phy": {
		"all": {
			"path": "$PARAMS_FOLDER/lussac/final_output",
			"wvf_extraction": {
				"ms_before": 1.0,
				"ms_after": 3.0,
				"max_spikes_per_unit": 1000
			},
			"export_params": {
				"compute_amplitudes": true,
				"compute_pc_features": false,
				"copy_binary": false,
				"template_mode": "average",
				'sparsity': {
					"method": "radius",
					"radius_um": 75.0
				},
				"verbose": False
			},
			"estimate_contamination": {
				"all": [0.3, 1.0]
			}
		}
	}


The :code:`export_to_sigui` module
----------------------------------

| This module will export all sortings in their current state to the SpikeInterface GUI format (if :code:`merge_sortings` was called before, will only export the merged sorting).
| This is equivalent to just a :code:`WaveformExtractor` with some extra arguments.

This module's parameters are:

- :code:`path`: path to the folder where to export the sorting(s). If multiple sortings exists, a subfolder will be created for each of them.
- :code:`wvf_extraction`: to construct the templates.
- :code:`spike_amplitudes` (optional): either a :code:`dict` or :code:`False`. If a :code:`dict`, will compute and export the spike amplitudes, the content of the dictionary being the parameters for :code:`spikeinterface.postprocessing.compute_spike_amplitudes`. By default :code:`dict()`.
- :code:`principal_components` (optional): either a :code:`dict` or :code:`False`. If a :code:`dict`, will compute and export the PCA, the content of the dictionary being the parameters for :code:`spikeinterface.postprocessing.compute_principal_components`. By default :code:`False`.


Example of export to SI GUI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

	"export_to_sigui": {
		"all": {
			"path": "$PARAMS_FOLDER/lussac/final_output",
			"wvf_extraction": {
				"ms_before": 1.0,
				"ms_after": 3.0,
				"max_spikes_per_unit": 1000
			}
		}
	}
