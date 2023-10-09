Installation
============

:code:`lussac` is a python package that needs to be installed.
It is recommended to install :code:`lussac` in an environment, such as conda:

.. code-block:: bash

	conda create -n lussac python=3.11  # Must be >=3.10
	conda activate lussac


|:spider_web:| Installing from PyPi
-----------------------------------

You can install :code:`lussac` directly from PyPi using pip:

.. code-block:: bash

	pip install lussac
	# pip install --upgrade lussac  # To upgrade in case a new version is released.


|:construction_site:| Installing from source
--------------------------------------------

You can install :code:`lussac` from source by cloning the GitHub repository (developmental version). This will ensure you have the latest updates, but some features might still be experimental!

.. code-block:: bash

	# Download Lussac in any directory you want.
	git clone https://github.com/BarbourLab/lussac.git --branch dev
	cd lussac

	# Install Lussac.
	pip install -e .[dev]

	# For the developmental version, you will likely need the latest developmental version of SpikeInterface
	git clone https://github.com/SpikeInterface/spikeinterface.git
	cd spikeinterface
	pip install -e .

	# If you want to check whether the installation was successful (optional)
	pytest
