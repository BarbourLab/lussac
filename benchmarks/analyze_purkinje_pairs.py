import json
import os
from pathlib import Path

import jsmin
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from lussac.core import LussacData, LussacPipeline
import probeinterface.io
import spikeinterface.core as si
from spikeinterface.core.testing import check_sortings_equal
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.sorters as ss


folder = Path("purkinje_pairs/")
os.makedirs(folder, exist_ok=True)

data_folder = Path(f"TODO")

if not data_folder.exists():
	print("Server has not been mounted!\nCan't access it.")
	exit(1)

n_purk = {}

mice_ID = sorted([path.name for path in data_folder.glob("0*")])
for mouse_ID in tqdm(mice_ID):
	mouse_folder = data_folder / mouse_ID

	if mouse_ID == "0142":
		continue  # Problem with its different sampling frequency
	
	eyeblink_sessions = list(mouse_folder.glob("*_eyeblink_*/rec_*"))
	if len(eyeblink_sessions) == 0:
		continue

	# Loading the recording
	try:  # If we can load from local, then we don't need to copy
		recording = si.load_extractor(folder / mouse_ID / "recording.dat")
		mouse_folder = mouse_folder / recording.get_annotation("session_name")
	except:
		sessions_id = [session_name.parent.name.split('_')[-1] for session_name in eyeblink_sessions]
		first_eyeblink_index = np.argsort(np.array(sessions_id, dtype=np.int16))[0]

		mouse_folder = mouse_folder / eyeblink_sessions[first_eyeblink_index].parent

		with open(mouse_folder / "lussac.json") as file:
			params = json.load(file)['recording']['extractor_params']

		probe_group = probeinterface.io.read_probeinterface(mouse_folder.parent / "prb_spikeinterface")
		recording = si.BinaryRecordingExtractor(mouse_folder / "extraCellAnalysis/phy/recording.dat", sampling_frequency=params['sampling_frequency'], dtype=params['dtype'],
												num_channels=params['num_chan'], gain_to_uV=params['gain_to_uV'], offset_to_uV=params['offset_to_uV'])
		recording = recording.set_probegroup(probe_group)
		recording.annotate(mouse_ID=mouse_ID)
		recording.annotate(session_name=eyeblink_sessions[first_eyeblink_index].parent.name)

		recording = recording.save(folder=folder / mouse_ID / "recording.dat")

	# Loading the analyses
	sortings = {}
	with open(mouse_folder / "lussac.json") as file:
		analyses = json.load(file)['analyses']
		analyses = {name: mouse_folder / path[15:] for name, path in analyses.items()}

	for name, path in analyses.items():
		try:  # If we can load it from local, then we don't need to copy
			sortings[name] = si.load_extractor(folder / mouse_ID / "analyses" / name)
		except:
			if os.path.exists(path / "spike_times.npy"):
				sorting = se.PhySortingExtractor(path)
			else:
				sorting = si.load_extractor(path)

			sorting.annotate(analysis_name=name)
			sortings[name] = sorting.save(folder=folder / mouse_ID / "analyses" / name)

	# Running Lussac
	try:
		sortings['lussac'] = se.PhySortingExtractor(folder / mouse_ID / "analyses" / "lussac2" / "phy_final")
	except:
		with open(mouse_folder / "lussac.json") as file:
			minified = jsmin.jsmin(file.read())
			minified = minified.replace("$PARAMS_FOLDER", str(folder / mouse_ID / "analyses"))
			lussac_params = json.loads(minified)

			# Changes in the params that need to be done due to newer versions of SpikeInterface / Lussac
			del lussac_params['lussac']['si_global_job_kwargs']['verbose']
			lussac_params['lussac']['pipeline']['merge_sortings']['CS']['correlogram_validation'] = dict(max_time=300.0, gaussian_std=7.0)
			lussac_params['lussac']['pipeline']['merge_sortings']['CS']['waveform_validation'] = {'waveform_extraction': dict(ms_before=1.0, ms_after=3.0, filter_band=[150, 6_000.0])}
			lussac_params['lussac']['pipeline']['merge_sortings']['CS']['merge_check'] = dict(cross_cont_threshold=0.5)
			lussac_params['lussac']['pipeline']['merge_sortings']['CS']['clean_edges'] = dict(template_diff_threshold=0.15, corr_diff_threshold=0.20, cross_cont_threshold=0.25)

		lussac_data = LussacData(recording, sortings, lussac_params)
		lussac_pipeline = LussacPipeline(lussac_data)
		lussac_pipeline.launch()

		sortings['lussac'] = se.PhySortingExtractor(folder / mouse_ID / "analyses" / "lussac2" / "phy_final")

	# Running default ks 2
	try:
		sortings['ks2_def'] = se.PhySortingExtractor(folder / mouse_ID / "analyses" / "ks2_default" / "sorter_output")
	except:
		sortings['ks2_def'] = ss.run_sorter(sorter_name="kilosort2", recording=recording, folder=folder / mouse_ID / "analyses" / "ks2_default", singularity_image=True, remove_existing_folder=True,
								delete_tmp_files=["matlab_files", "temp_wh.dat"], delete_recording_dat=True, extra_requirements=["numpy==1.26"])

	# Running optimal ks 2
	try:
		sortings['ks2_opt'] = se.PhySortingExtractor(folder / mouse_ID / "analyses" / "ks2_optimal" / "sorter_output")
	except:
		recording_tmp = spre.filter(recording, band=[280, 6000], filter_order=2, ftype="bessel")
		sortings['ks2_opt'] = ss.run_sorter(sorter_name="kilosort2", recording=recording_tmp, folder=folder / mouse_ID / "analyses" / "ks2_optimal", singularity_image=True, remove_existing_folder=True,
								projection_threshold=[8, 3], freq_min=40, delete_tmp_files=["matlab_files", "temp_wh.dat"], delete_recording_dat=True, extra_requirements=["numpy==1.26"])

	# Counting the number of Purkinje cells
	for name, sorting in sortings.items():
		try:
			analyzer = si.load_sorting_analyzer(folder / mouse_ID / "analyses" / name / "analyzer")
			assert analyzer.has_extension("correlograms")
		except:
			analyzer = si.create_sorting_analyzer(sorting, recording, format="binary_folder", folder=folder / mouse_ID / "analyses" / name / "analyzer", overwrite=True, sparse=False, return_scaled=True)
			analyzer.compute("correlograms", window_ms=25.0, bin_ms=1.0, method="numba")
		
		firing_rates = sqm.compute_firing_rates(analyzer)

		putative_ss_units = [unit_id for unit_id, mean_fr in firing_rates.items() if mean_fr >= 40.0]
		putative_cs_units = [unit_id for unit_id, mean_fr in firing_rates.items() if 0.5 <= mean_fr <= 3.0]

		# sorting = sorting.select_units([*putative_ss_units, *putative_cs_units])
		correlograms, bins = analyzer.get_extension("correlograms").get_data()
		mask = ((bins >= 0.0) & (bins < 8.0))[:-1]
		n_pairs = 0

		for ss_id in putative_ss_units:
			for cs_id in putative_cs_units:
				ss_ind = analyzer.sorting.id_to_index(ss_id)
				cs_ind = analyzer.sorting.id_to_index(cs_id)
				cross_corr = correlograms[ss_ind, cs_ind, :]

				baseline = np.median(cross_corr[bins[:-1] < 0.0])
				if (np.median(cross_corr[mask]) < 0.4*baseline):  # Check for pause
					if np.median(cross_corr[mask[::-1]] < 0.4*baseline):  # Check for asymmetry
						continue
					n_pairs += 1  # TODO: Check for templates
					break

		if name in n_purk:
			n_purk[name] += n_pairs
		else:
			n_purk[name] = n_pairs

print(n_purk)

fig = go.Figure()

fig.add_trace(go.Bar(
	x=list(n_purk.keys()),
	y=list(n_purk.values())
))

fig.show()
