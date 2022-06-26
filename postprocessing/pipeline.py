import os
import time
import math
import numpy as np
import spiketoolkit

from phy_data import PhyData
import postprocessing.center_cluster as center_cluster
import postprocessing.center_waveform as center_waveform
import postprocessing.merge_sortings as merge_sortings
import postprocessing.merge_units as merge_units
import postprocessing.remove_bad_units as remove_bad_units
import postprocessing.remove_duplicates as remove_duplicates
import postprocessing.utils as utils


methods = {
	"center_waveform": center_waveform.center_units,
	"center_cluster": center_cluster.center_units,
	"merge_sortings": merge_sortings.automerge_sortings,
	"merge_clusters": merge_units.automerge_units,
	"remove_duplicates": remove_duplicates.remove_duplicated_spikes,
	"remove_bad_units": remove_bad_units.rm_bad_units
}


def launch(data: PhyData, params: dict, prb_file: str):
	"""
	Launches the post-processing pipeline to curate all sortings in data
	function of what 'params' tells us to do.

	@param data (PhyData):
		The data object.
	@param params (dict):
		Parameters for the pipeline.
	"""

	for func, params0 in params.items():
		plot_folder = "{0}/{1}".format(data.logs_folder, func)

		if func.split('_')[-1].isnumeric(): # Allows to put the same function multiple times in the dict.
			func = '_'.join(func.split('_')[:-1])

		if func.split('_')[0] == "export":
			params0['prb'] = prb_file

		if func == "units_categorization":
			potential_units = [dict() for i in range(data.get_num_sortings())]

			for sorting in range(data.get_num_sortings()):
				data.set_sorting(sorting)
				potential_units[sorting] = categorize_units(data, params0, dict())[0]

		elif func == "units_recategorization":
			for sorting in range(data.get_num_sortings()):
				data.set_sorting(sorting)
				potential_units[sorting], removed_units = categorize_units(data, params0, potential_units[sorting])

				for category, units in removed_units.items():
					leftout = params0[category]['leftout']

					if leftout[0] == '#':
						if leftout == "#rm":
							data.sorting.exclude_units(units)
						else:
							assert False, "Error in units_recategorization.{0}.leftout: command \"{1}\" not found".format(category, leftout)
					elif leftout not in potential_units:
						potential_units[leftout] = units
					else:
						assert False, "Error in units_recategorization.{0}.leftout: category \"{1}\" already exists".format(category, leftout)

		elif func == "merge_sortings":
			_print_section(func)

			if data.get_num_sortings() <= 1:
				print("\t- Aborted: You need more than 1 sorting to use merge_sortings.")

			for category, params1 in params0.items():
				units_id = dict()
				for sorting in range(data.get_num_sortings()):
					data.set_sorting(sorting)
					units_id[sorting] = _get_units_by_category(data, potential_units[sorting], category)

					if len(units_id[sorting]) == 0:
						del units_id[sorting]

				new_units_ids = _launch_section(merge_sortings.automerge_sortings, category, data, units_id, params1, plot_folder)

			data.merged_sorting = spiketoolkit.curation.CurationSortingExtractor(data.merged_sorting)
			for root in data.merged_sorting._roots:
				root.set_spike_train(root.get_spike_train().astype(np.uint64))
			data._sortings = [data.merged_sorting]
			data.set_sorting(0)
			
		elif func == "export":
			_print_section(func)

			if len(data.merged_sorting.get_unit_ids()) == 0:
				print("\t- Aborting export: No units in the merged_sorting.")

			data.recording = data.recording.load_probe_file(params0['prb'])

			export_params = dict(compute_pc_features=True, compute_amplitudes=True, max_channels_per_template=data.recording.get_num_channels(), max_spikes_per_unit=None,
														copy_binary=False, ms_before=1.0, ms_after=3.0, dtype=np.int16, recompute_info=True, n_jobs=3, filter_flag=False)
			export_params.update(params0['export_params'])

			t1 = time.perf_counter()
			spiketoolkit.postprocessing.export_to_phy(data.recording, data.merged_sorting, params0['path'], **export_params)
			t2 = time.perf_counter()
			print("export_to_phy took {0:.0f} s".format(t2-t1))

			cell_types = "cluster_id\tcell_type"
			contamination = "cluster_id\tContPct"
			previous_ID = "cluster_id\tprevID"
			i = 0
			for unit_id in data.merged_sorting.get_unit_ids():
				category = "MS"
				for cat, units in potential_units[0].items():
					if unit_id in units:
						category = cat
						break

				refractory_period = tuple(params0['refractory_period'][category])
				C = utils.estimate_unit_contamination(data, unit_id=unit_id, refractory_period=refractory_period)

				cell_types += "\n{0}\t{1}".format(i, category)
				contamination += "\n{0}\t{1:.2f}".format(i, 100*C)
				previous_ID += "\n{0}\t{1}".format(i, unit_id)
				i += 1

			f = open("{0}/cell_type.tsv".format(params0['path']), "w+")
			f.write(cell_types)
			f.close()
			f = open("{0}/cont_pct.tsv".format(params0['path']), "w+")
			f.write(contamination)
			f.close()
			f = open("{0}/prev_unit_id.tsv".format(params0['path']), "w+")
			f.write(previous_ID)
			f.close()

		elif func == "export_sortings":
			_print_section(func)

			recording = data.recording.load_probe_file(params0['prb'])

			for sorting in range(len(data._sortings)):
				data.set_sorting(sorting)
				if len(data.sorting.get_unit_ids()) == 0:
					print(f"\t- Sorting {sorting}: Aborted (no units to export)")

				folder = "{0}/sorting_{1}".format(params0['path'], sorting)
				os.makedirs(folder, exist_ok=True)

				export_params = dict(compute_pc_features=False, compute_amplitudes=False, max_channels_per_template=data.recording.get_num_channels(), max_spikes_per_unit=5000,
														copy_binary=False, ms_before=1.0, ms_after=3.0, dtype=np.int16, recompute_info=True, n_jobs=3, filter_flag=False)
				export_params.update(params0['export_params'])
				spiketoolkit.postprocessing.export_to_phy(recording, data._sortings[sorting], folder, **export_params)

				cell_types = "cluster_id\tcell_type"
				contamination = "cluster_id\tContPct"
				previous_ID = "cluster_id\tprevID"
				i = 0
				for unit_id in data._sortings[sorting].get_unit_ids():
					category = "MS"
					for cat, units in potential_units[sorting].items():
						if unit_id in units:
							category = cat
							break

					refractory_period = tuple(params0['refractory_period'][category])
					C = utils.estimate_unit_contamination(data, unit_id=unit_id, refractory_period=refractory_period)

					cell_types += "\n{0}\t{1}".format(i, category)
					contamination += "\n{0}\t{1:.2f}".format(i, 100*C)
					previous_ID += "\n{0}\t{1}".format(i, unit_id)
					i += 1

				f = open("{0}/cell_type.tsv".format(folder), "w+")
				f.write(cell_types)
				f.close()
				f = open("{0}/cont_pct.tsv".format(folder), "w+")
				f.write(contamination)
				f.close()
				f = open("{0}/prev_unit_id.tsv".format(folder), "w+")
				f.write(previous_ID)
				f.close()

				print(f"\t- Sorting {sorting}: OK")

		else:
			_print_section(func)

			for sorting in range(data.get_num_sortings()):
				print("Sorting {0}:".format(sorting))
				data.set_sorting(sorting)

				for category, params1 in params0.items():
					if ('sortings' in params1) and (sorting not in params1['sortings']):
						continue

					units_id = _get_units_by_category(data, potential_units[sorting], category)
					if len(units_id) == 0:
						continue

					new_units_ids = _launch_section(methods[func], category, data, units_id, params1, plot_folder)
					if new_units_ids != None:
						_update_units_id(data, potential_units[sorting], category, new_units_ids)


def categorize_units(data: PhyData, params: dict, potential_units: dict):
	"""
	Categorizes units in different categories based on some parameters.
	TODO: duplicate units if appears in multiple categories.

	@param data (PhyData):
		The data object.
	@param params (dict):
		Parameters for categorization.
	@param potential_units (dict):
		Categories that already exist (leaving it blank will populate the category from scratch).

	@return potential_units (dict):
		New potential units based on the given parameters.
	@return removed_units (dict):
		Units removed from the given potential_units.
	"""

	removed_units = dict()
	for category, rules in params.items():
		removed_units[category] = []

		if not category in potential_units:
			potential_units[category] = list(data.sorting.get_unit_ids())
		else:
			potential_units[category] = list(potential_units[category])

		if 'frequency' in rules and isinstance(rules['frequency'], list) and len(rules['frequency']) == 2:
			for unit_id in potential_units[category][::-1]:
				firing_rate = len(data.get_unit_spike_train(unit_id)) / data.recording.get_num_frames() * data.sampling_f

				if firing_rate < rules['frequency'][0] or firing_rate > rules['frequency'][1]:
					potential_units[category].remove(unit_id)
					removed_units[category].append(unit_id)

		if 'contamination' in rules and isinstance(rules['contamination'], dict):
			for unit_id in potential_units[category][::-1]:
				contamination = utils.estimate_unit_contamination(data, unit_id=unit_id, refractory_period=tuple(rules['contamination']['refractory_period']))

				if 'max' in rules['contamination'] and contamination > rules['contamination']['max']:
					potential_units[category].remove(unit_id)
					removed_units[category].append(unit_id)
				if 'min' in rules['contamination'] and contamination < rules['contamination']['min']:
					potential_units[category].remove(unit_id)
					removed_units[category].append(unit_id)

		if 'suppression_period' in rules and isinstance(rules['suppression_period'], dict):
			assert 'contamination' in rules, "Used suppression_period in categorization without contamination first. Category {0}".format(category)
			for unit_id in potential_units[category][::-1]:
				contamination = utils.estimate_unit_contamination(data, unit_id=unit_id, refractory_period=tuple(rules['contamination']['refractory_period']))
				suppression_period = utils.get_unit_supression_period(data, unit_id, contamination)

				if 'max' in rules['suppression_period'] and suppression_period > rules['suppression_period']['max']:
					potential_units[category].remove(unit_id)
					removed_units[category].append(unit_id)
				if 'min' in rules['suppression_period'] and suppression_period < rules['suppression_period']['min']:
					potential_units[category].remove(unit_id)
					removed_units[category].append(unit_id)

		if 'ISI' in rules and isinstance(rules['ISI'], dict):
			for unit_id in potential_units[category][::-1]:
				bin_size = rules['ISI']['bin_size'] if "bin_size" in rules['ISI'] else 1.0
				ISI = utils.get_ISI(data, unit_id=unit_id, bin_size=bin_size, max_time=rules['ISI']['range'][1])[0]
				ISI = np.sum(ISI[int(round(rules['ISI']['range'][0] / bin_size)):]) / len(data.sorting.get_unit_spike_train(unit_id))

				if "min" in rules['ISI'] and ISI < rules['ISI']['min']:
					potential_units[category].remove(unit_id)
					removed_units[category].append(unit_id)
				elif "max" in rules['ISI'] and ISI > rules['ISI']['max']:
					potential_units[category].remove(unit_id)
					removed_units[category].append(unit_id)

		if 'duplicate' in rules and isinstance(rules['duplicate'], dict):
			for unit_id in potential_units[category][::-1]:
				for cat, allowed in rules['duplicate'].items():
					if unit_id in potential_units[cat] and not allowed:
						potential_units[category].remove(unit_id)
						removed_units[category].append(unit_id)

		potential_units[category] = np.array(potential_units[category], dtype=np.uint16)

	return potential_units, removed_units


def _print_section(name: str, n_stars: int=35):
	"""
	Prints that we enter into a section in the console.

	@param name (str):
		Section's name.
	@param n_stars (int):
		How many stars/characters per line.
	"""

	half = (n_stars - 2 - len(name)) / 2

	print("\n" + "*"*n_stars)
	print("{0} {1} {2}".format("*"*math.floor(half), name, "*"*math.ceil(half)))
	print("*"*n_stars)


def _launch_section(func: callable, name: str, data: PhyData, units: list, params: dict, plot_folder: str):
	"""
	Launches a section, prints the time it took, and return the new_units_ids.

	@param func (callable):
		Function to call.
	@param name (str):
		Name to display in the print and used for the plot folder.
	@param data (PhyData):
		The data object.
	@param units (list or np.ndarray):
		The units' id to pass on in the function.
	@param params (dict):
		The params to params to pass on in the function.
	@param plot_folder (str):
		Path to the function folder for plots.

	@return new_units_ids (TODO):
		TODO
	"""

	t1 = time.perf_counter()
	new_units_ids = func(data, units, params, plot_folder="{0}/sorting_{1}/{2}".format(plot_folder, data.sorting_idx, name)) # TODO: check if folder already exists. If yes, increment center_waveform_2
	t2 = time.perf_counter()
	print("\t- {0} ({1:.1f} s)".format(name, t2-t1))

	return new_units_ids


def _get_units_by_category(data: PhyData, potential_units: dict, category: str):
	"""
	Returns the units' id that correspond to a category.

	@param data (PhyData):
		The data object.
	@param potential_units (dict):
		Dict containing the categories and units by category.
	@param category (str):
		Category to extract. Must be "all", "rest" or must be in potential_units.
		"all" extracts all of the units' id, no matter if they are in a category or not.
		"rest" extractsall of the units' id that are not in a category.

	@return units_id np.ndarray [n_units]:
		Array containing the units' id that correspond to the given category.
	"""

	if ';' in category:
		units_id = list()

		for cat in category.split(';'):
			units_id.extend(list(_get_units_by_category(data, potential_units, cat)))

		return np.array(units_id)

	if category in potential_units:
		return potential_units[category]
	if category == "all":
		return data.sorting.get_unit_ids()
	if category == "rest":
		units_id = list(data.sorting.get_unit_ids())

		for cat, ids in potential_units.items():
			units_id = [unit for unit in units_id if unit not in ids]

		return np.array(units_id)

	assert False, "Did not understand category \"{0}\"".format(category)


def _update_units_id(data: PhyData, potential_units: dict, category: str, new_units_ids: dict):
	"""
	TODO: remake this function.
	"""

	if ';' in category:
		for cat in category.split(';'):
			_update_units_id(data, potential_units, cat, new_units_ids)

		return

	if category == "rest":
		return
	if category == "all":
		for old_id, new_id in new_units_ids.items():
			for cat in potential_units:
				if not old_id in potential_units[cat]:
					continue

				idx = np.argmax(potential_units[cat] == old_id)

				if new_id == -1: # Deleted unit
					potential_units[cat] = np.delete(potential_units[cat], idx)
				else:
					potential_units[cat][idx] = new_id

				potential_units[cat] = np.unique(potential_units[cat])
	else:
		for old_id, new_id in new_units_ids.items():
			if not old_id in potential_units[category]:
				continue

			idx = np.argmax(potential_units[category] == old_id)

			if new_id == -1: # Deleted unit
				potential_units[category] = np.delete(potential_units[category], idx)
			else:
				potential_units[category][idx] = new_id

		potential_units[category] = np.unique(potential_units[category])
