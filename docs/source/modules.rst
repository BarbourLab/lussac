Lussac modules
==============

Here, we describe each of Lussac modules, what they do, how they do it, and what parameters are available.


How waveforms are extracted
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many different modules, waveforms need to be extracted a :code:`wvf_extraction` parameter dictionary is used to know how waveforms should be extracted and processed.
The different parameters are:

- :code:`ms_before`: how many samples (in ms) should be extracted before the spike peak.
- :code:`ms_after`: how many samples (in ms) should be extracted after the spike peak.
- :code:`max_spikes_per_unit`: how many spikes should be extracted per unit to create the template.
- :code:`filter_band`: a list of 2 floats, representing the minimum and maximum frequency (in Hz) for a Gaussian bandpass filter. *Note that Lussac extracts an extra margin to remove border filtering artifacts*.


The :code:`units_categorization` module
---------------------------------------

This module will label units as belonging to a certain category if they meet some criteria. If a unit already belongs to a category, it will not be re-categorized. So the order matters!

This module takes as a key the name of the category, and as a value a dictionary containing the criteria. Each criterion return a value for each unit, and a minimum and/or maximum can be set.

You can also specify the parameters for :code:`wvf_extraction` (`max_spikes_per_unit`, `ms_before`, `ms_after`, `filter`).

- :code:`firing_rate`: returns the mean firing rate of the unit (in Hz).
- :code:`contamination`: returns the estimated contamination of the unit (between 0 and 1; 0 being pure). The :code:`refractory_period = [censored_period, refractory_period]` has to be set (in ms).
- :code:`amplitude`: returns the mean amplitude of the unit's template (in µV if the recording object can be scaled to µV). Optional parameters can be set to the function :code:`spikeinterface.core.get_template_extremum_amplitude`.
- :code:`SNR`: returns the signal-to-noise ratio of the unit. Optional parameters can be set to the function :code:`spikeinterface.qualitymetrics.compute_snrs`
- :code:`sd_ratio`: returns the standard deviation in the spike amplitudes for the unit divided by the standard deviation on the same channel. Optional parameters can be set to the function :code:`spikeinterface.postprocessing.compute_spike_amplitudes` as a dictionary with key :code:`spike_amplitudes_kwargs`. Optional parameters can be set to the function :code:`spikeinterface.qualitymetrics.compute_sd_ratio` as a dictionary with key :code:`sd_ratio_kwargs`.
- :code:`ISI_portion`: Returns the fraction (between 0 and 1) of inter-spike intervals inside a time range. the :code:`range = [min_t, max_t]` has to be set (in ms).


Example of categorization
^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example for categorizing complex-spikes from more regular spikes (cerebellar cortex example):

.. code-block:: json

    "units_categorization": {
        "all": {  // Categorize all units.
            "wvf_extraction": {  // Parameters for the waveform extraction.
                "ms_before": 1.0,
                "ms_after": 1.5,
                "max_spikes_per_unit": 500,
                "filter": [150.0, 7000.0]  // Gaussian bandpass filter with cutoffs at 150 and 7,000 Hz.
            },
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
                "max_spikes_per_unit": 2000,  // Use 2,000 random spikes to construct templates.
                "filter_band": [300.0, 6000.0]  // Gaussian-filter between 300 and 6000 Hz.
            },
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
            "wvf_extraction": {},  // If you want to change how the waveforms are extracted.
            "contamination": {
                "refractory_period": [1.5, 25.0],
                "max": 0.35
            }
        },
        "spikes": {  // Remove units with firing rate < 1.0 Hz or amplitude std > 80 µV
            "firing_rate": {
                "min": 1.0
            },
            "sd_ratio": {
                "max": 2.0
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
                "remove_strategy": "highest_amplitude"  // Keep the unit with the highest amplitude.
            }
        }
    }


The :code:`merge_units` module
------------------------------

This module looks for units that correspond to the same neuron (inside each individual analysis separately), and merges them together if the merge is deemed beneficial.

| This is done by first looking over all pairs of units, and estimating if they likely come from the same neuron, on the basis of: proximity, matching correlograms, matching templates.
| Then, pairs that don't increase the quality score if the merge is performed are discarded. With this discard, the worse of both units is removed (because it usually is a bad split unit).
| Finally, a graph is constructed from the remaining pairs. For each connected component (i.e. each putative neuron), we iteratively merge the best pair until everything is merged or there are no more merges that increase the quality score metric. If some unmerged units remain, they are discarded.

This modules parameters are:

* :code:`refractory_period = [censored_period, refractory_period]`: in ms. By default: :code:`[0.2, 1.0]`.

* :code:`wvf_extraction`: to construct the templates.

* :code:`correlograms`: a :code:`dict` containing the parameters to construct the correlograms.

  * :code:`window_ms`: The **total** window size of the correlogram (in ms). A value of :code:`100.0` will create a correlogram of size :code:`[-50.0, 50.0]` ms. By default: 150 ms.

  * :code:`bin_ms`: The size of the bins in the correlogram (in ms). By default: 0.04 ms.

* :code:`auto_merge_params`: a :code:`dict` containing the parameters to give to :code:`spikeinterface.curation.auto_merge.compute_merge_unit_groups`.


Example of merging units
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    "merge_units": {
        "all": {
            "refractory_period": [0.2, 1.0],
            "wvf_extraction": {
                "ms_before": 1.0,
                "ms_after": 1.5,
                "max_spikes_per_unit": 2000,
                "filter_band": [150, 7000]
            },
            "correlograms": {
                "window_ms": 150,
                "bin_ms": 0.04
            },
            "auto_merge_params": {
                "steps_params": {
                    "correlogram": {
                        "corr_diff_thresh": 0.16,
                        "censor_correlograms_ms": 0.2,
                        "sigma_smooth_ms": 0.6,
                        "adaptive_window_thresh": 0.5
                    },
                    "template_similarity": {"template_diff_thresh": 0.25}
                },
                "firing_contamination_balance": 2.5,  // k = 2.5 in the paper.
                "resolve_graph": false  // False by default because Lussac implements its own graph resolution.
            }
        }
    }


The :code:`merge_sortings` module
---------------------------------

This module is the heart of Lussac. It will merge all individual analyses into a single one, following a complex algorithm.

- :code:`STEP 1`: Create a graph where each node is a unit, and each edge links similar units (based on the correlation of their spike trains).
- :code:`STEP 2`: Detect and remove merged units.
- :code:`STEP 3`: Detect "wrong" edges and remove them.
- :code:`STEP 4`: For each community, create/select the best unit.

The parameters used in this module are:

- :code:`refractory_period = [censored_period, refractory_period]`: in ms. By default: :code:`[0.2, 1.0]`.

- :code:`max_shift`: The maximum time shift when re-aligning pairs of units (in ms). By default: 1.33 ms.

- :code:`require_multiple_sortings_match`: Whether to remove lone units (i.e. units that are not matched with any other unit). By default: True.

- :code:`similarity`: a :code:`dict` to compute the similarity (i.e. spike trains correlation) in STEP 1.

  - :code:`min_similarity`: The minimum similarity to consider two units similar. By default: 0.3.

  - :code:`window`: The maximum lag (in ms) allowed between two spikes to be considered similar. By default: 0.2 ms.

- :code:`correlogram_validation`: a :code:`dict` to compute the validation correlogram in STEP 3.

  - :code:`max_time`: The maximum time for the correlogram (in ms). By default: 70 ms (i.e. correlogram computed between :code:`[-70, 70]` ms).

  - :code:`gaussian_std`: The standard deviation of the Gaussian kernel used to smooth the correlogram (in ms). By default: 0.6 ms.

  - :code:`gaussian_truncate`: The Gaussian is truncated after X standard deviations for faster computation. By default: X = 5.

  - :code:`bin_ms` (optional): The size of the bins in the correlogram (in ms). By default: very small, adaptative to :code:`max_time`.

- `waveform_validation`: a :code:`dict` to compute the validation waveform in STEP 3.

  - :code:`wvf_extraction`: to construct the templates. By default :code:`ms_before = 1.0`, :code:`ms_after = 2.0`, :code:`max_spikes_per_unit = 1000`, :code:`filter_band = [250, 6000]`.

  - :code:`num_channels`: The number of channels used to compare waveforms. By default: 5.

- :code:`merge_check`: a :code:`dict` to compute the merge check in STEP 2.

  - :code:`cross_cont_threshold`: The threshold above which the cross-contamination is considered too high. By default: 0.10. *Note that the cross-contamination needs to be significantly higher, using a statistical test*.

- `clean_edges`: a :code:`dict` with the thresholds used for STEP 3.

  - :code:`template_diff_threshold`: The threshold above which the template difference is considered too high. By default: 0.10.

  - :code:`corr_diff_threshold`: The threshold above which the correlation difference is considered too high. By default: 0.12.

  - :code:`cross_cont_threshold`: The threshold above which the cross-contamination is considered too high. By default: 0.06. *Note that the cross-contamination needs to be significantly higher, using a statistical test*.


Example of merging sortings
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    "merge_sortings": {
        "all": {
            "refractory_period": [0.2, 1.0],
            "max_shift": 1.33,
            "require_multiple_sortings_match": true,
            "similarity": {
                "min_similarity": 0.3,
                "window": 0.2
            },
            "correlogram_validation": {
                "max_time": 70.0,
                "gaussian_std": 0.6,
                "gaussian_truncate": 5.0
            },
            "waveform_validation": {
                "wvf_extraction": {
                    "ms_before": 1.0,
                    "ms_after": 2.0,
                    "max_spikes_per_unit": 1000,
                    "filter_band": [250.0, 6000.0]
                },
                "num_channels": 5
            },
            "merge_check": {
                "cross_cont_threshold": 0.10
            },
            "clean_edges": {
                "template_diff_threshold": 0.10,
                "corr_diff_threshold": 0.12,
                "cross_cont_threshold": 0.06
            }
        }
    }


The :code:`find_purkinje_cells` module
--------------------------------------

| This module is only meant for cerebellar cortex recordings. It will link simple spikes and complex spikes coming from the same Purkinje cell, and set it as a property :code:`lussac_purkinje` (this property is automatically exported in the :code:`export_to_phy` module).
| TODO: Explain how it works.

This module's parameters are:

- :code:`cross_corr_pause`: the band over which to look for the pause (in ms). By default: :code:`[0.0, 8.0]`.
- :code:`threshold`: TODO
- :code:`ss_min_fr`: Minimum firing rate to consider putative simple spikes (in Hz). By default: :code:`40.0`.
- :code:`cs_min_fr`: Minimum firing rate to consider putative complex spikes (in Hz). By default: :code:`0.5`.
- :code:`cs_max_fr`: Maximum firing rate to consider putative complex spikes (in Hz). By default: :code:`3.0`.


Example of finding Purkinje cells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    "find_purkinje_cells": {
        "all": {
            "cross_corr_pause": [0.0, 8.0],
            "threshold": 0.4,
            "ss_min_fr": 40.0,
            "cs_max_fr": 3.0
        }
    }


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
                "sparsity": {
                    "method": "radius",
                    "radius_um": 75.0
                },
                "verbose": false
            },
            "estimate_contamination": {
                "all": [0.3, 1.0]
            }
        }
    }


The :code:`export_to_sigui` module
----------------------------------

| This module will export all sortings in their current state to the SpikeInterface GUI format (if :code:`merge_sortings` was called before, will only export the merged sorting).
| This is equivalent to just a :code:`SortingAnalyzer` with some extra arguments.

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
