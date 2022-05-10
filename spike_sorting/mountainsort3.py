import os
import shutil
import csv
import json
import subprocess
import numpy as np
import spikeextractors
import spiketoolkit

from .spike_sorter import SpikeSorter


class MountainSort3(SpikeSorter):

	def launch(self, name: str, params: dict={}):
		"""

		"""

		if os.path.exists(self.output_folder + "/output"):
			assert False, "Error: output folder '{0}/output' already exists, and would have been overwritten!".format(output_folder)

		tmp_folder = "{0}/tmp".format(self.output_folder)
		output_folder = "{0}/{1}".format(self.output_folder, name)

		os.makedirs(tmp_folder, exist_ok=True)
		os.makedirs(output_folder, exist_ok=True)
		os.makedirs("{0}/spk_interface".format(tmp_folder), exist_ok=True)

		self._create_mda(tmp_folder)
		self._create_geom(tmp_folder)
		self._create_params(tmp_folder, params)
		self._create_pipeline(tmp_folder)
		output, error = self._run_ms3(tmp_folder)

		# TODO: check that firings.mda was created. If not, mountainsort3 failed (on my computer, it randomly fails 1/20th of the time).

		with open("{0}/logs.txt".format(output_folder), 'w') as logs_file:
			if output != None:
				logs_file.write("Output:\n")
				logs_file.write(output.decode(encoding="utf-8"))
			if error != None:
				logs_file.write("\n\n\nError:\n")
				logs_file.write(error.decode(encoding="utf-8"))

		sorting = spikeextractors.MdaSortingExtractor(file_path="{0}/firings.mda".format(tmp_folder), sampling_frequency=self.data_params['sampling_rate'])
		sorting.set_tmp_folder("{0}/spk_interface".format(tmp_folder))
		spiketoolkit.postprocessing.export_to_phy(self.recording, sorting, output_folder, compute_pc_features=False, compute_amplitudes=False, max_channels_per_template=self.data_params['n_channels'],
														max_spikes_per_unit=1000, copy_binary=False, ms_before=1.0, ms_after=3.0, dtype=self.data_params['dtype'], recompute_info=True, n_jobs=3, filter_flag=False)

		del sorting
		shutil.rmtree(tmp_folder)


	def _create_mda(self, tmp_folder: str):
		"""
		Saves the data in mda format to be understood by MountainSort 3.
		"""

		filename = "data.mda"
		data = self.recording.get_traces()
		data = data - np.median(data, axis=0)[None, :]

		# Information about mda file format can be found at: https://mountainsort.readthedocs.io/en/latest/mda_file_format.html
		mda_header = np.zeros([3 + len(data.shape)], dtype=np.int32)

		if self.data_params['dtype'] == "float32" or self.data_params['dtype'] == "float":
			mda_header[0] = -3
			mda_header[1] = 4
		elif self.data_params['dtype'] == "int16":
			mda_header[0] = -4
			mda_header[1] = 2
		elif self.data_params['dtype'] == "int32":
			mda_header[0] = -5
			mda_header[1] = 4
		elif self.data_params['dtype'] == "uint16":
			mda_header[0] = -6
			mda_header[1] = 2
		elif self.data_params['dtype'] == "double" or self.data_params['dtype'] == "float64":
			mda_header[0] = -7
			mda_header[1] = 8
		elif self.data_params['dtype'] == "uint32":
			mda_header[0] = -8
			mda_header[1] = 4

		mda_header[2] = len(data.shape)
		mda_header[3:] = data.shape

		file = open("{0}/{1}".format(tmp_folder, filename), "wb")
		mda_header.T.astype(np.int32).tofile(file)
		file.close()
		file = open("{0}/{1}".format(tmp_folder, filename), "ab")
		data.T.astype(self.data_params['dtype']).tofile(file)
		file.close()


	def _create_geom(self, tmp_folder: str):
		"""
		Saves the geometry of the electrode in a format understood by MountainSort 3.
		"""

		channel_ids = self.recording.get_channel_ids()
		n_channels = len(channel_ids)
		n_dim = len(self.recording.get_channel_property(channel_ids[0], "location"))
		geom = np.zeros([n_channels, n_dim])

		for i in range(n_channels):
			geom[i, :] = self.recording.get_channel_property(channel_ids[i], "location")

		with open("{0}/geom.csv".format(tmp_folder), 'w') as file:
			writer = csv.writer(file, delimiter='\t', lineterminator='\n')

			for row in geom:
				writer.writerow(row)


	def _create_params(self, tmp_folder: str, params: dict):
		"""
		Saves the parameters as a JSON for MountainSort 3.
		"""

		with open("{0}/params.json".format(tmp_folder), 'w') as file:
			json.dump(params, file)


	def _create_pipeline(self, tmp_folder: str):
		"""
		Creates the file to tell the MountainSort 3 pipeline what to do.
		"""

		text = """\
				{"processing_server":"typhoon","pipelines":[{"spec":{"name":"main","description":"","inputs":[],"outputs":[],"parameters":[]},"steps":[{"step_type":"pipeline","pipeline_name":"synthesize","inputs":{},"outputs":{"raw":"raw","geom":"geom","waveforms_true":"waveforms_true","firings_true":"firings_true"},"parameters":{"duration":"600","samplerate":"30000"}},{"step_type":"pipeline","pipeline_name":"sort","inputs":{"raw":"raw","geom":"geom"},"outputs":{"firings_out":"firings","filt_out":"filt","pre_out":"pre"},"parameters":{"samplerate":"30000","detect_sign":""}},{"step_type":"pipeline","pipeline_name":"curate","inputs":{"pre":"pre","firings":"firings"},"outputs":{"curated_firings":"curated_firings"},"parameters":{"samplerate":"30000"}}],"input_files":[],"output_files":[]},{"spec":{"name":"synthesize","description":"","inputs":[],"outputs":[{"name":"raw"},{"name":"geom"},{"name":"waveforms_true"},{"name":"firings_true"}],"parameters":[{"name":"duration","description":"Durations of simulated dataset in seconds"},{"name":"samplerate"}]},"steps":[{"step_type":"processor","processor_name":"pyms.synthesize_random_waveforms","inputs":{},"outputs":{"waveforms_out":"waveforms_true","geometry_out":"geom"},"parameters":{"upsamplefac":"13"}},{"step_type":"processor","processor_name":"pyms.synthesize_random_firings","inputs":{},"outputs":{"firings_out":"firings_true"},"parameters":{"samplerate":"${samplerate}","duration":"${duration}"}},{"step_type":"processor","processor_name":"pyms.synthesize_timeseries","inputs":{"firings":"firings_true","waveforms":"waveforms_true"},"outputs":{"timeseries_out":"raw"},"parameters":{"duration":"${duration}","waveform_upsamplefac":"13"}},{"step_type":"processor","processor_name":"pyms.synthesize_random_firings","inputs":{},"outputs":{"firings_out":"test_firings","console_out":"test_cons"},"parameters":{"duration":"310"}}],"input_files":[],"output_files":[]},{"name":"sort","script":"/* Define the spec */\ninputs_opt('raw','filt','pre','geom');\noutputs('firings_out');\noutputs_opt('filt_out','pre_out','firings_original_out');\nparam('samplerate',30000);\nparam('freq_min',300);\nparam('freq_max',6000);\nparam('freq_wid',1000);\nparam('whiten','true');\nparam('detect_threshold',3);\nparam('detect_sign',0);\nparam('adjacency_radius',-1);\nparam('curate','false');\nparam('consolidation_factor',0.9);\nparam('detect_interval', 5);\nparam('clip_size',50);\nparam('fit_stage','true');\n\n_Pipeline.run=function(X) {\n  var pp=X.parameters;\n  \n  var pre='pre';\n  if (!X.hasInput('pre')) {\n    \n    var filt='filt';\n    if (!X.hasInput('filt')) {\n      if (!X.hasInput('raw')) {\n        console.error('Missing input: raw, filt or pre');\n        return -1;\n      }\n      X.runProcess('ms3.bandpass_filter',\n                   {timeseries:'raw'},\n                   {timeseries_out:'filt_out'},\n                   {samplerate:pp.samplerate,freq_min:pp.freq_min,freq_max:pp.freq_max,freq_wid:pp.freq_wid}\n                  );\n      filt='filt_out';\n    }\n  \n  \n    if (pp.whiten=='true') {\n      X.runProcess('ms3.whiten',\n                   {timeseries:filt},\n                   {timeseries_out:'pre_out'},\n                   {}\n                  );\n    }\n    else {\n      X.runProcess('pyms.normalize_channels',\n                   {timeseries:'filt'},\n                   {timeseries_out:'pre_out'},\n                   {}\n                  );\n    }\n    pre='pre_out';\n  }\n  \n  \n  var curate=(pp.curate=='true');\n  var firings1='firings_out';\n  if (curate) firings1='firings_original_out';\n  \n  var p={\n    detect_threshold:pp.detect_threshold,\n    detect_sign:pp.detect_sign,\n    adjacency_radius:pp.adjacency_radius\n,\n    clip_size:pp.clip_size\n,\n    detect_interval:pp.detect_interval\n,\n    consolidation_factor:pp.consolidation_factor\n,\n    fit_stage:pp.fit_stage\n  };\n  var inputs={timeseries:pre};\n  if (X.hasInput('geom')) {\n    inputs.geom='geom';\n  }\n  X.runProcess('mountainsortalg.ms3',\n               inputs,\n               {firings_out:firings1},\n               p);\n  \n  if (curate) {\n    X.runPipeline('curate',\n               {pre:pre,firings:firings1},\n               {curated_firings:'firings_out'},\n               {samplerate:pp.samplerate});\n  }\n               \n};\n\n/////////////////////////////////////////////////////////////////////\n\n\nfunction param(str,val) {\n      if (val===undefined) {\n        _Pipeline.spec.parameters.push({name:str});\n      }\n      else {\n        _Pipeline.spec.parameters.push({name:str,optional:true,default_value:val});\n      }\n}\n                \nfunction inputs(str1,str2,str3,str4) {\n  if (str1) _Pipeline.spec.inputs.push({name:str1});\n  if (str2) _Pipeline.spec.inputs.push({name:str2});\n  if (str3) _Pipeline.spec.inputs.push({name:str3});\n  if (str4) _Pipeline.spec.inputs.push({name:str4});\n}\n\nfunction inputs_opt(str1,str2,str3,str4) {\n  if (str1) _Pipeline.spec.inputs.push({name:str1,optional:true});\n  if (str2) _Pipeline.spec.inputs.push({name:str2,optional:true});\n  if (str3) _Pipeline.spec.inputs.push({name:str3,optional:true});\n  if (str4) _Pipeline.spec.inputs.push({name:str4,optional:true});\n}\n\nfunction outputs(str1,str2,str3,str4) {\n  if (str1) _Pipeline.spec.outputs.push({name:str1});\n  if (str2) _Pipeline.spec.outputs.push({name:str2});\n  if (str3) _Pipeline.spec.outputs.push({name:str3});\n  if (str4) _Pipeline.spec.outputs.push({name:str4});\n}\n\nfunction outputs_opt(str1,str2,str3,str4) {\n  if (str1) _Pipeline.spec.outputs.push({name:str1,optional:true});\n  if (str2) _Pipeline.spec.outputs.push({name:str2,optional:true});\n  if (str3) _Pipeline.spec.outputs.push({name:str3,optional:true});\n  if (str4) _Pipeline.spec.outputs.push({name:str4,optional:true});\n}","steps":[],"spec":{"name":"","description":"","inputs":[],"outputs":[],"parameters":[]},"export":true},{"spec":{"name":"curate","description":"","inputs":[{"name":"pre"},{"name":"firings"}],"outputs":[{"name":"curated_firings"}],"parameters":[{"name":"samplerate"}]},"steps":[{"step_type":"processor","processor_name":"ms3.cluster_metrics","inputs":{"timeseries":"pre","firings":"firings"},"outputs":{"cluster_metrics_out":"metrics1"},"parameters":{"samplerate":"${samplerate}"}},{"step_type":"processor","processor_name":"ms3.isolation_metrics","inputs":{"timeseries":"pre","firings":"firings"},"outputs":{"metrics_out":"metrics2"},"parameters":{"compute_bursting_parents":"true"}},{"step_type":"processor","processor_name":"ms3.combine_cluster_metrics","inputs":{"metrics_list":["metrics1","metrics2"]},"outputs":{"metrics_out":"metrics"},"parameters":{}},{"step_type":"processor","processor_name":"pyms.create_label_map","inputs":{"metrics":"metrics"},"outputs":{"label_map_out":"label_map"},"parameters":{}},{"step_type":"processor","processor_name":"pyms.apply_label_map","inputs":{"firings":"firings","label_map":"label_map"},"outputs":{"firings_out":"curated_firings"},"parameters":{}}],"input_files":[],"output_files":[]}],"input_files":[],"output_files":[],"jobs":[]}\
				"""
		text = text.replace("\n", "\\n")

		with open("{0}/mountainsort3.mlp".format(tmp_folder), 'w') as file:
			file.write(text)


	def _run_ms3(self, tmp_folder: str):
		"""
		Runs the bash command for MountainSort 3.
		"""

		bash = "mlp-run mountainsort3.mlp sort --raw=data.mda --geom=geom.csv --firings_out=firings.mda --_params=params.json".format(tmp_folder)

		process = subprocess.Popen(bash.split(), cwd=tmp_folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()

		return output, error
