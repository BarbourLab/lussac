import copy
from pathlib import Path
import numpy.typing as npt
import plotly.graph_objects as go
import spikeinterface.core as si


def _plot_sliders(fig: go.Figure, traces_per_plot: int, labels: npt.ArrayLike, filepath: str, args: dict | None = None, plots_per_file: int = 30) -> None:
	"""
	Takes a figure with multiple traces and plots, and decomposes it with a slider for each plot.
	If there are too many plots, will split the plots into multiple files.

	@param fig: go.Figure
		The plotly figure object.
	@param traces_per_plot: int
		Number of traces inside each plot.
	@param labels: ArrayLike
		The labels for each plot.
	@param filepath: str
		The path to the file to save the plot.
		Needs to be without any extension!
	@param args: dict | None
		The arguments of what needs to change when the slider is moved.
	@param plots_per_file: int
		The maximum number of plots per file.
	"""
	if args is None:
		args = {}
	Path(filepath).parent.mkdir(parents=True, exist_ok=True)

	N = len(labels)
	n_files = 1 + (N - 1) // plots_per_file

	for n in range(n_files):
		start = (n * N) // n_files
		end = ((n+1) * N) // n_files

		sub_fig = copy.deepcopy(fig)
		sub_fig.data = sub_fig.data[traces_per_plot*start : traces_per_plot * end]
		for i in range(len(sub_fig.data)):
			sub_fig.data[i].visible = i < traces_per_plot

		steps = []
		for i in range(end - start):
			steps.append({
				'label': f"{labels[start+i]}",
				'method': "update",
				'args': [
					{'visible': [j // traces_per_plot == i for j in range(traces_per_plot * (end-start))]},
					args
				]
			})

		sub_fig.layout['sliders'] = [{
			'active': 0,
			'currentvalue': {'prefix': 'Unit: '},
			'pad': {'t': 50},
			'steps': steps
		}]

		filename = f"{filepath}_{n+1}.html" if n_files > 1 else f"{filepath}.html"
		sub_fig.write_html(filename)


def plot_units(wvf_extractor: si.WaveformExtractor, filepath: str, n_channels: int = 3) -> None:
	"""
	Plots all the units in a given sorting.

	@param wvf_extractor: si.WaveformExtractor
		The waveform extractor object containing the units to plot.
	@param filepath: str
		The path to the file to save the plot.
		Needs to be without any extension!
	"""

	fig = go.Figure().set_subplots(rows=2+(n_channels-1)//3, cols=3)

	# TODO: Do stuff

	_plot_sliders(fig, 3+n_channels, labels=wvf_extractor.unit_ids, filepath=filepath)
