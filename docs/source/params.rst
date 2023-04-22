Lussac parameters
=================

Each run of Lussac relies on a :code:`params.json` configuration file. This fully-customizable file contains all the parameters required to read your data and run the desired modules.

You can either write your own, or start with one provided in the :code:`params_examples/` folder (note that you **must** still change the :code:`recording` section to accommodate your data).

Lussac's pipeline relies on "modules". The structure of the :code:`params.json` file dictates which modules are run, in which order and with which parameters (note that a module can be run multiple times at different stages of the pipeline).
A module will perform a specific task, such as removing bad units, or merging units, or removing duplicated spikes ...

Lussac also relies heavily on `SpikeInterface <https://github.com/SpikeInterface/spikeinterface>`_, a Python framework designed to unify preexisting spike sorting technologies in a single code base. Knowing SpikeInterface isn't required to run Lussac, but it can help.


The structure of the :code:`params.json` file
---------------------------------------------

The file :code:`params.json` contains all of the Lussac parameters that you can modify to control your custom electrophysiological analysis. It is divided into 4 sections:

- :code:`recording`: containing the necessary information for Lussac to read your electrophysiological recordings.
- :code:`spike_sorting`: containing the information for Lussac to automatically run the spike sorting algorithms (optional).
- :code:`analyses`: containing the paths to the phy output folders containing the already spike-sorted data (if you chose not to run spike-sorting through Lussac). (Note that if Lussac ran a spike-sorter in the previous section, the output will automatically be appended in this section).
- :code:`lussac`: containing the information required for Lussac to run your desired post-processing and/or merging of multiple analyses.

If you are using one of the provided :code:`params.json` file, then you will **need** to update the :code:`recording` section so that Lussac can understand your data. The other sections can be left unmodified, or can be adjusted to your preferences.

More in-depth information about each section is given below.


The :code:`json` language
^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration file is written in `JSON <https://en.wikipedia.org/wiki/JSON>`_, a popular and open-source data-interchange file format that has the advantage to be human-readable (even with little programming experience). It is also easily translated to Python.

Curly brackets in JSON are translated to a :code:`dict` in Python. Hence, the two following codes (in json and python) are equivalent:

.. code-block:: json

	{
		"key1": {
			"foo": "bar",
			"index": 4
		}
		"key2": true,
		"key3": 3.141592653589793
	}

.. code-block:: python

	dict(
		key1=dict(foo="bar", index=4),
		key2=True,
		key3=3.141592653589793
	)

- You can type in :code:`$PARAMS_FOLDER` anywhere, and it will be replaced with the path to the folder containing the :code:`params.json` file (useful to copy paste the json config file to another folder).
- You can add comments in C style (using :code:`//` for inline comment, and :code:`/* */` for multiline comments). This is **not** in the JSON language by default, but has been added here because it can be useful.


The :code:`recording` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section contains the information for Lussac to read the electrophysiological data (such as sampling rate, number of channels, etc.). The recording is loaded using `SpikeInterface <https://github.com/SpikeInterface/spikeinterface>`_, so every format supported by SpikeInterface is also supported by Lussac.

There are multiple keys:

- :code:`recording_extractor`: the name of the recording extractor in SpikeInterface.
- :code:`extractor_params`: the parameters for the recording extractor.
- :code:`probe_file`: the path to the probe file (in the `ProbeInterface <https://github.com/SpikeInterface/probeinterface>`_ format) if not already in the recording (optional).

Example for a binary file:
""""""""""""""""""""""""""

.. code-block:: json

	"recording": {
		"recording_extractor": "BinaryRecordingExtractor",
		"extractor_params": {
			"file_path": "$PARAMS_FOLDER/recording.dat",
			"numchan": 64,
			"sampling_frequency": 30000,
			"dtype": "int16",
			"gain_to_uV": 0.195,
			"offset_to_uV": 0.0
		},
		"probe_file": "$PARAMS_FOLDER/probe.json"

Example for a SpikeGLX recording:
"""""""""""""""""""""""""""""""""

.. code-block:: json

	"recording": {
		"recording_extractor": "SpikeGLXRecordingExtractor",
		"extractor_params": {
			"folder_path": "$PARAMS_FOLDER/recording",
			"stream_id": "imec0.ap"
		}
		// Probe is already loaded with the SpikeGLXRecordingExtractor.
	}


The :code:`spike_sorting` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section contains the information for Lussac to run the spike sorting algorithms (optional). You can also not include this section and instead provide yourself the analyses in the :code:`analyses` section.

| If you want to run spike sorting through Lussac, you will need to either have the spike sorters installed in the same environment, or you can have :code:`docker` or :code:`singularity` installed to run the spike sorters in a container.
| SpikeInterface allows you to run sorters in a **container**, which can be really neat. You'll need to install either :code:`docker` or :code:`singularity` (which can take a while), but once installed you'll have access to a lot of spike sorters without needing to have them installed (and without requiring matlab!)
| See the `SpikeInterface documentation <https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html#running-sorters-in-docker-singularity-containers>`_ on the installation if you are interested.
| For Linux user, we recommend installing `singularity` as it is easier than docker to deal with root access.

To run sorters, the :code:`spike_sorting` section is made like this:

- A :code:`dict` mapping the run name to another :code:`dict`, containing:
	- :code:`sorter_name`: the name of the sorter in SpikeInterface.
	- :code:`preprocessing`: (optional) a :code:`dict` mapping a function in `spikeinterface.preprocessing` to a :code:`dict` containing the arguments for that function.
	- :code:`sorter_params`: the parameters for the sorter.

Example for running 2 spike sorters
"""""""""""""""""""""""""""""""""""

The following code will run kilosort 3 (with singularity) and SpykingCircus (installed locally):

.. code-block:: json

	"spike_sorting": {
		"ks3_sing": {  // Kilosort 3 analysis using singularity and some custom parameters.
			"sorter_name": "kilosort3",
			"preprocessing": {
				"filter": {"band": [300., 6000.], "filter_order": 2, "ftype": "bessel"},  // Custom bessel filter
				"common_reference": {"operator": "median"}  // Common median reference.
			},
			"sorter_params": {
				"output_folder": "$PARAMS_FOLDER/analyses/ks3_sing",
				"singularity_image": true,
				"projection_threshold": [8, 8],  // Lower Kilosort's threshold.
				"freq_min": 40  // Filter already applied in preprocessing.
			}
		},
		"sc_default": {  // Spyking Circus analysis using the default parameters.
			"sorter_name": "spykingcircus",
			"sorter_params": {
				"output_folder": "$PARAMS_FOLDER/analyses/sc_default"
			}
		}
	}


The :code:`analyses` section
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section contains the already spike-sorted analyses you want to feed Lussac. The analyses must be in the `Phy <https://github.com/cortex-lab/phy>`_ format (if they are not, you can use SpikeInterface :code:`export_to_phy`).

This :code:`dict` maps the analysis name to its location. For example:

.. code-block:: json

	"analyses": {
		"ks2_default": "path/to/ks2_analysis",
		"tdc_default": "path/to/tridesclous_analysis"
	}


The :code:`lussac` section
^^^^^^^^^^^^^^^^^^^^^^^^^^

This section contains all the information needed for Lussac to know what to do with your data (i.e. post-processing and merging of multiple analyses). It is divided into 4 keys:

- :code:`logs_folder`: the path where to store the logs for Lussac (you will be able to inspect what Lussac did in this folder). If the directory doesn't exist, Lussac will create it. If the directory already exists and contains information about a previous run, Lussac will load this information (if a previous run crashed, Lussac will pick up where it left off).
- :code:`tmp_folder`: the path to the temporary directory. To not load everything in memory, Lussac needs to write some information on the disk (preferentially a fast SSD rather than an HDD). The directory will be created by Lussac and removed at the end of the run.
- :code:`si_global_job_kwargs`: some global keyword arguments for SpikeInterface (such as number of jobs, chunking ...). See example below.
- :code:`pipeline`: a dictionary containing what modules to run and in which order. See the next section below.


Typical structure for the :code:`lussac` section
================================================

.. code-block:: json

	"lussac": {
		"logs_folder": "$PARAMS_FOLDER/lussac/logs",
		"tmp_folder": "$PARAMS_FOLDER/lussac/tmp",
		"si_global_job_kwargs": {
			"n_jobs": 6,  // Number of threads to use on the CPU. Can be increased or decreased depending on your computer.
			"chunk_duration": "2s",
			"progress_bar": false,
			"verbose": false
		},
		"pipeline": {
			/** "first_module_name": {module_1_params},
				"second_module_name: {module_2_params},
			   ...
			*/
		}
	}


Lussac module system
--------------------

Work in progress...
