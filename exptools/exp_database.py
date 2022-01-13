""" A script package helping you make plots for paper.
"""
import os, sys, json
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from contextlib import redirect_stderr
from os import devnull

from exptools.launching.variant import flatten_variant4hparams

class ExpDatabase:
	""" A comparison tool that help you get all data paths that match your report needs
	"""
	def __init__(self,
			exp_paths,
			config_filename= "variant_config.json",
			required_files = [],
		):
		"""
		@ Args:
			exp_paths: list[str] absolute paths
			config_filename: the config file that used to check if the experiment is needed
			required_files: a list of strings that are the required files to be in the experiment directory
				NOTE: You cannot specify the files that are in the subdirectory of your experiment directory
		"""
		self.exp_paths = exp_paths
		self.config_filename = config_filename
		self.required_files = required_files
		self.color_map = dict()
		self.df = pd.DataFrame()
		
		self.interpret_exp_paths()

	def interpret_exp_paths(self):
		""" Go through all paths, and log them into a pandas data frame arranged by params
		"""
		# extract all valid log directory
		for exp_path in self.exp_paths:
			for path, subdirs, subfiles in os.walk(exp_path):
				if self.config_filename in subfiles and all([f in subfiles for f in self.required_files]):
					# get a valid directory
					with open(os.path.join(path, self.config_filename)) as f:
						param = json.load(f)
					# to make all dictionary in the same level, which is easier to process
					param = flatten_variant4hparams(param)
					abspath = os.path.abspath(path)
					param["experiment_log_path"] = abspath
					self.df = self.df.append(param, ignore_index= True)
		if self.df.empty:
			Warning("You got an empty database, please check your exp_paths")

	def get_datapaths(self, configs: dict):
		""" This method will return all experiments paths that match the configs will be collected and returned.
		NOTE: This method will return a list of paths, not a pandas dataframe
		Args
			configs: a dict whose key is a string, value is a list of config values
		Return
			a list of tuple
				(0): each combination of configs (specified by configs)
				(1): a list of all experiment's directory (who satisfies the configs specification)
					(n): the directory of the experiment you want to collect
		"""

		# fill all configs which is assigned as None, and seperate them
		for key, options in configs.items():
			assert isinstance(options, list) or options is None, "You must provide a list of options you want, or simply None"
			if options is None:
				configs[key] = [*(self.df[key].unique())]
		keys = list(configs.keys())
		all_options = list(configs.values()) # a list of options (list)

		return_ = []
		for i, all_option in enumerate(itertools.product(*all_options)):
			# all_option is a list of experiment value
			df = self.df
			for key, option in zip(keys, all_option):
				df = df.loc[df[key] == option]
			data_paths = df["experiment_log_path"]
			
			if len(data_paths) == 0:
				Warning("No experiment found for {}".format(all_option))
				continue
			else:
				return_.append((
					dict(zip(keys, all_option)), # dict
					data_paths, # list
				))
		return return_
