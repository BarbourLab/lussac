import os
import pathlib
import numpy as np
import plotly.graph_objects as go
from lussac.core import LussacData
import lussac.utils as utils
import spikeinterface.core as si


folder = "tests/tmp/plotting"


def test_get_path_to_plotlyJS() -> None:
	plotly_js = utils.Utils.plotly_js_file

	path = plotly_js.parent / "aze" / "dfg"
	assert utils.get_path_to_plotlyJS(path) == pathlib.Path("../../plotly.min.js")
	assert utils.get_path_to_plotlyJS(str(path)) == pathlib.Path("../../plotly.min.js")


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
	sorting.set_property("gt_label", np.array([f"label{unit_id}" for unit_id in sorting.unit_ids]))
	analyzer = si.create_sorting_analyzer(sorting, data.recording, format="binary_folder", folder="tests/tmp/plotting/analyzer", sparse=False)
	analyzer.compute({
		'random_spikes': {'max_spikes_per_unit': 100},
		'fast_templates': {'ms_before': 1.5, 'ms_after': 2.5},
	})

	utils.plot_units(analyzer, filepath=f"{folder}/plot_units", annotations_fix=[{'text': "I am a fixed annotation"}],
					 annotations_change=[{'text': f"I am unit {unit_id}"} for unit_id in analyzer.unit_ids])
	utils.plot_units(analyzer, filepath=f"{folder}/plot_units_all_channels", n_channels=100000)
	assert os.path.exists(f"{folder}/plot_units.html")
	assert os.path.exists(f"{folder}/plot_units_all_channels.html")


def test_create_gt_annotations() -> None:
	sorting = si.NumpySorting.from_unit_dict({0: np.array([3, 60], dtype=np.int64), 1: np.array([90, 187, 601], dtype=np.int64)}, 30000)
	assert len(utils.create_gt_annotations(sorting)) == 0

	sorting.set_property(key="gt_label", values=['Good', 'Bad'])
	assert len(utils.create_gt_annotations(sorting)) == 2
