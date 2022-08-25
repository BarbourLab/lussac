import setuptools
import lussac.version


def open_requirements(filename: str) -> list[str]:
	"""
	Opens the requirements file and returns its content.

	@param filename: str
		Path to the requirements file.
	@return requirements: list[str]
		List of all the requirements.
	"""

	with open(filename, mode='r') as requirements_file:
		requirements = requirements_file.read().splitlines()

	return [requirement for requirement in requirements if len(requirement) > 0 and not requirement.startswith('#')]


setuptools.setup(
	name="lussac",
	version=lussac.version.version,
	author="Aurélien Wyngaard, Victor Llobet, Boris Barbour",
	author_email="boris.barbour@ens.fr",
	description="Python package for automatic post-processing and merging of multiple spike-sorting analyses.",
	url="https://github.com/BarbourLab/lussac",
	install_requires=open_requirements("requirements.txt"),
	entry_points={'console_scripts': "lussac=lussac.main:main"}
)
