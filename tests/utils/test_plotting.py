import os
import numpy as np
import plotly.graph_objects as go
import spikeinterface.core as si
from lussac.core.lussac_data import LussacData
import lussac.utils as utils


folder = "tests/tmp/plotting"


def test_get_path_to_plotlyJS() -> None:
	plotly_js = LussacData.PLOTLY_JS

	path = plotly_js.parent / "aze" / "dfg"
	assert utils.get_path_to_plotlyJS(path) == "../../plotly.min.js"


def test_plot_sliders() -> None:
	fig = go.Figure().set_subplots(rows=1, cols=2)
	xaxis = np.arange(0, 2*np.pi, 1e-2)

	for i in range(10):
		fig.add_trace(go.Scatter(
			x=xaxis,
			y=np.sin(2*np.pi*i * xaxis),
			mode="lines",
			name=f"sin freq={i} Hz"
		), row=1, col=1)
		fig.add_trace(go.Scatter(
			x=xaxis,
			y=np.exp(-i * xaxis),
			mode="lines",
			name=f"exp tau={i} s"
		), row=1, col=2)

	fig.update_xaxes(title_text="Time (s)")
	fig.update_yaxes(title_text="Amplitude")

	utils.plot_sliders(fig, 2, labels=list(range(10)), filepath=f"{folder}/test_sliders", plots_per_file=20)
	utils.plot_sliders(fig, 2, labels=list(range(10)), filepath=f"{folder}/test_sliders", plots_per_file=5)

	assert os.path.exists(f"{folder}/test_sliders.html")
	assert os.path.exists(f"{folder}/test_sliders_1.html")
	assert os.path.exists(f"{folder}/test_sliders_2.html")
	assert not os.path.exists(f"{folder}/test_sliders_3.html")


def test_plot_units(data: LussacData) -> None:
	sorting = data.sortings['ks2_best'].select_units([13, 19, 40, 41])
	wvf_extractor = si.extract_waveforms(data.recording, sorting, folder="tests/tmp/plotting/wvf_extractor", ms_before=1.5, ms_after=2.5, max_spikes_per_unit=500)

	utils.plot_units(wvf_extractor, filepath=f"{folder}/plot_units")
	utils.plot_units(wvf_extractor, filepath=f"{folder}/plot_units_all_channels", n_channels=100000)
	assert os.path.exists(f"{folder}/plot_units.html")
	assert os.path.exists(f"{folder}/plot_units_all_channels.html")
