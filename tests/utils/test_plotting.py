import os
import numpy as np
import plotly.graph_objects as go
import lussac.utils.plotting as plotting


def test_plot_sliders() -> None:
	folder = "tests/tmp/plotting"
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

	plotting._plot_sliders(fig, 2, labels=list(range(10)), filepath=f"{folder}/test_sliders", plots_per_file=20)
	plotting._plot_sliders(fig, 2, labels=list(range(10)), filepath=f"{folder}/test_sliders", plots_per_file=5)

	assert os.path.exists(f"{folder}/test_sliders.html")
	assert os.path.exists(f"{folder}/test_sliders_1.html")
	assert os.path.exists(f"{folder}/test_sliders_2.html")
	assert not os.path.exists(f"{folder}/test_sliders_3.html")
