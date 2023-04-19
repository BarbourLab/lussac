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
			"gain_to_uV": 0.195
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

Work in progress ...
