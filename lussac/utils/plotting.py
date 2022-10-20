import os
import copy
from pathlib import Path
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import spikeinterface.core as si
import spikeinterface.postprocessing as spost
from lussac.core.lussac_data import LussacData
import lussac.utils as utils


def get_path_to_plotlyJS(path: str | Path) -> str:
	"""
	Returns the path to the plotly.js file relative to the given path.

	@param path: str | Path
		The path to where you are.
	@return plotlyJS_path: str
		The path to plotly.js relative to the given path.
	"""
	if isinstance(path, str):
		path = Path(path)

	return os.path.relpath(LussacData.PLOTLY_JS, start=path)


def plot_sliders(fig: go.Figure, traces_per_plot: int, labels: npt.ArrayLike, filepath: str, args: list[dict] | None = None, plots_per_file: int = 30) -> None:
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
	@param args: list[dict] | None
		The arguments of what needs to change when the slider is moved.
	@param plots_per_file: int
		The maximum number of plots per file.
	"""
	if args is None:
		args = [{}] * len(labels)
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
					args[start+i]
				]
			})

		sub_fig.layout['sliders'] = [{
			'active': 0,
			'currentvalue': {'prefix': 'Unit: '},
			'pad': {'t': 50},
			'steps': steps
		}]

		sub_fig.update_layout(**args[start])

		filename = f"{filepath}_{n+1}.html" if n_files > 1 else f"{filepath}.html"
		sub_fig.write_html(filename, include_plotlyjs=get_path_to_plotlyJS(Path(filename).parent))


def plot_units(wvf_extractor: si.WaveformExtractor, filepath: str, n_channels: int = 3, max_time_ms: float = 35., bin_size_ms: float = 0.25,
			   firing_rate_std: float = 5.) -> None:
	"""
	Plots all the units in a given sorting.

	@param wvf_extractor: si.WaveformExtractor
		The waveform extractor object containing the units to plot.
	@param filepath: str
		The path to the file to save the plot.
		Needs to be without any extension!
	@param n_channels: int
		The number of channels to plot for the template.
	@param max_time_ms: float
		The maximum time for ISI and auto-correlogram plots (in ms).
	@param bin_size_ms: float
		The bin size for ISI and auto-correlogram plots (in ms).
	@param firing_rate_std: float
		The standard deviation for the gaussian smoothing of the firing rate.
	"""
	sf = wvf_extractor.sampling_frequency
	max_time = int(round(max_time_ms * 1e-3 * sf))
	bin_size = int(round(bin_size_ms * 1e-3 * sf))
	xaxis = (np.arange(wvf_extractor.nsamples) - wvf_extractor.nbefore) / sf * 1e3

	if n_channels > wvf_extractor.recording.get_num_channels():
		n_channels = wvf_extractor.recording.get_num_channels()

	fig = go.Figure().set_subplots(rows=2+(n_channels-1)//3, cols=3)
	args = []

	for unit_id in wvf_extractor.unit_ids:
		spike_train = wvf_extractor.sorting.get_unit_spike_train(unit_id)
		args.append({
			"title.text": f"Unit {unit_id}"
		})

		ISI, bins = spost.compute_isi_histograms_from_spiketrain(spike_train, max_time, bin_size, sf)
		bin = (bins[1] - bins[0]) / 2
		fig.add_trace(go.Bar(
			x=bins[:-1] + bin,
			y=ISI,
			width=2*bin,
			name="ISI",
			marker_color="CornflowerBlue"
		), row=1, col=1)

		auto_corr = spost.compute_autocorrelogram_from_spiketrain(spike_train, max_time, bin_size)
		bins, _, _ = spost.correlograms._make_bins(wvf_extractor.sorting, 2*max_time_ms, bin_size_ms)
		bin = (bins[1] - bins[0]) / 2
		fig.add_trace(go.Bar(
			x=bins[:-1] + bin,
			y=auto_corr,
			width=2*bin,
			name="Auto-correlogram",
			marker_color="CornflowerBlue"
		), row=1, col=2)

		taxis = np.arange(0, wvf_extractor.recording.get_num_samples() + sf/2, sf / 2)
		firing_rate = utils.gaussian_histogram(spike_train, taxis, sigma=firing_rate_std * sf, margin_reflect=True) * sf
		fig.add_trace(go.Scatter(
			x=taxis / sf,
			y=firing_rate,
			mode="lines",
			name="Firing rate (Hz)",
			marker_color="CornflowerBlue"
		), row=1, col=3)

		template = wvf_extractor.get_template(unit_id, mode="average")
		best_channels = np.argsort(np.max(np.abs(template), axis=0))[::-1]
		for i in range(n_channels): # TODO: share y axis for all templates in a unit.
			channel = best_channels[i]
			fig.add_trace(go.Scatter(
				x=xaxis,
				y=template[:, channel],
				mode="lines",
				name=f"Template channel {wvf_extractor.channel_ids[channel]}",
				marker_color="CornflowerBlue"
			), row=2 + i//3, col=1 + i%3)

	fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
	fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
	fig.update_xaxes(title_text="Time (s)", row=1, col=3)
	fig.update_yaxes(title_text="Firing rate (Hz)", rangemode="tozero", row=1, col=3)
	for i in range(n_channels):
		fig.update_xaxes(title_text="Time (ms)", row=2 + i//3, col=1 + i%3)
		fig.update_yaxes(title_text=f"Amplitude ({'µV' if wvf_extractor.return_scaled else 'A.U.'})", rangemode="tozero", row=2 + i//3, col=1 + i%3)

	plot_sliders(fig, 3 + n_channels, labels=wvf_extractor.unit_ids, filepath=filepath, args=args)
