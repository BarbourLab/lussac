import os
import glob
import numpy as np
from setuptools import setup
from Cython.Build import cythonize


directives = {
	"boundscheck": False,
	"wraparound": False,
	"overflowcheck": False,
	"cdivision": True,
	"cdivision_warnings": False,
	"language_level": '3'
}

setup(
	ext_modules = cythonize("postprocessing/utils.pyx", compiler_directives=directives),
	include_dirs=[np.get_include()]
)

name = glob.glob("utils.cp*")[0]
os.remove("postprocessing/utils.cpp")	# Can be useful for debugging.
os.rename(name, "postprocessing/{0}".format(name))
