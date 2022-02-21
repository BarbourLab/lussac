import os
import glob
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
	ext_modules = cythonize("postprocessing/utils.pyx", compiler_directives=directives)
)

name = glob.glob("utils.cpython*")[0]
os.rename(name, "postprocessing/{0}".format(name))
