Installation
============

:code:`lussac` is a python package that needs to be installed.
It is recommended to install :code:`lussac` in an environment, such as conda:

.. code-block:: bash

	conda create -n lussac python=3.11  # Must be >=3.10
	conda activate lussac


|:stop_sign:| Installing from PyPi
----------------------------------

|:warning:| This method is not available yet! |:warning:|


|:construction_site:| Installing from source
--------------------------------------------

You can install :code:`lussac` from source by cloning the GitHub repository. This will ensure you have the latest updates, but some features might still be experimental!

.. code-block:: bash

	# Download Lussac in any directory you want.
	git clone https://github.com/BarbourLab/lussac.git
	cd lussac

	# Install Lussac in your environment.
	pip install -e .
