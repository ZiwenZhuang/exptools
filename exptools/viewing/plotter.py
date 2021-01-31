""" A script package helping you make plots for paper.
"""
import os, sys, json
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from exptools.launching.variant import flatten_variant4hparams

# NOTE: this could limit the number of curves of your plot
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
]

class PaperCurvePlotter:
	def __init__(self, exp_paths,
			log_filename= "progress.csv",
			param_filename= "variant_config.json",
			n_fig_a_row= 4, one_fig_size= (5, 5),
			fontsize= 18, fig_name_shorten_level= 0,
			x_lim: tuple= (-5*1e6, 50*1e6), y_lim: tuple=(0.0, 1e3),
			show_legend= False,
		):
		"""
		@ Args:
			exp_paths: list[str] absolute paths
			n_fig_a_row: int How many figure you want to put in a row
			fig_name_shorten_level: int to make the figure title shorter, the label
				will ignore the first some.
		"""
		self.exp_paths = exp_paths
		self.log_filename = log_filename
		self.param_filename = param_filename
		self.n_fig_a_row = n_fig_a_row
		self.one_fig_size = one_fig_size
		self.fontsize = fontsize
		self.x_lim = x_lim
		self.y_lim = y_lim
		self.fig_name_shorten_level = fig_name_shorten_level
		self.show_legend = show_legend
		self.color_map = dict()
		self.df = pd.DataFrame()
		
		self.interpret_exp_paths()

	def interpret_exp_paths(self):
		""" Go through all paths, and log them into a pandas data frame arranged by params
		"""
		# extract all valid log directory
		for exp_path in self.exp_paths:
			for path, subdirs, subfiles in os.walk(exp_path):
				if self.log_filename in subfiles and self.param_filename in subfiles:
					# get a valid directory
					with open(os.path.join(path, self.param_filename)) as f:
						param = json.load(f)
					# to make all dictionary in the same level, which is easier to process
					param = flatten_variant4hparams(param)
					abspath = os.path.abspath(path)
					param["experiment_log_path"] = abspath
					self.df = self.df.append(param, ignore_index= True)
		if self.df.empty:
			Warning("You got an empty database, please theck your exp_paths")

	def make_plots(self, args_in_figures, args_in_series, x_key, y_key,
			xlabel= None, ylabel= None,
			sci_lim= (1, 1),
		):
		""" The main entrance ploting all experiments by design. The two arguments are specifying
		what experiments you want to plot. Both of them are dictionaries, whose keys are param keys
		and values are a list of values you want to plot. If providing None, we will dig out all
		options in this given key.
		@ Args:
			args_in_figures: dict{str: list} plot all experiments that meet the specifications
				(seperated in each figure)
			args_in_series: dict{str: list} plot all experiments that meet the specifications
				(seperated in each curve)
			sci_lim: tuple(int, int) demote the scientific notation of x and y, (exponent of 10)
		"""
		self.marked_labels = []

		n_figures = 1
		figure_keys, figure_all_options = [], []
		for key, options in args_in_figures.items():
			# options: list
			if options is None:
				options = [*(self.df[key].unique())]
			n_figures *= len(options)
			figure_all_options.append(options)
			figure_keys.append(key)

		# record mapping from label name to options, which makes a 2 level nested dict
		self.series_label_mapping = dict()
		series_keys, series_all_options = [], []
		for key, options in args_in_series.items():
			if options is None:
				options = [*(self.df[key].unique())]
			series_all_options.append(options)
			series_keys.append(key)
		for i, options in enumerate(itertools.product(*series_all_options)):
			label = ""
			optionset = dict()
			for key, option in zip(series_keys, options):
				label += str(key) + ":" + str(option) + ";"
				optionset[key] = option
			self.series_label_mapping[label] = optionset
			self.color_map[label] = color_defaults[i]
		
		fig, axs = self.create_figure(n_figures)

		# plot each figure one by one
		for ax, fig_optionset in zip(
				axs.flat, itertools.product(*figure_all_options)
			):
			# get all rows that meet the figure options
			fig_df = self.df
			fig_name = ""
			for fig_key, fig_opt in zip(figure_keys, fig_optionset):
				fig_df = fig_df.loc[fig_df[fig_key] == fig_opt]
				fig_key = fig_key.split(".", self.fig_name_shorten_level)[-1]
				fig_name += fig_key + ":" + str(fig_opt) + ";"
			# plot each curve
			for label, series_optionset in self.series_label_mapping.items():
				# get all rows that meet the series options
				ser_df = fig_df
				for ser_key, ser_opt in series_optionset.items():
					ser_df = ser_df.loc[ser_df[ser_key] == ser_opt]
				self.plot_exp(ax,
					ser_df["experiment_log_path"], x_key, y_key,
					label= label,
				)
			
			# warp up for this figure
			ax.grid(color= "gray", linewidth= 0.5)
			ax.tick_params(axis='both', labelsize=14)
			ax.set_title(fig_name, fontsize=16)
			ax.set_xlim(self.x_lim)
			ax.ticklabel_format(style='sci', axis='x', scilimits=sci_lim)

		self.finish_plot(xlabel, ylabel)

	def finish_plot(self, xlabel, ylabel):
		if xlabel is not None: plt.xlabel(xlabel, fontsize= self.fontsize)
		if ylabel is not None: plt.ylabel(ylabel, fontsize= self.fontsize)
		plt.xlim(self.x_lim)
		plt.ylim(self.y_lim)
		plt.subplots_adjust(left= 0.05, bottom= 0.16)

		if self.show_legend:
			plt.subplots_adjust(right= 0.85)
			plt.legend(
				loc='upper left',
				bbox_to_anchor= (1, 1),
				handles= [mlines.Line2D([],[], color= v, label= k) for k, v in self.color_map.items()]
			)

		save_name = ylabel + "_to_" + xlabel + "_plots.png"
		save_filename = os.path.join(os.getcwd(), save_name)

		print("image saved at {}".format(save_filename))
		plt.savefig(save_filename)

	def create_figure(self, n_figures):
		legend_margin = 5 if self.show_legend else 0
		if n_figures > 1:
			n_cols = min(n_figures, self.n_fig_a_row)
			n_rows = (n_figures-1) // self.n_fig_a_row + 1
			fig, axs = plt.subplots(n_rows, n_cols,
				sharex= True,
				figsize= (n_cols*self.one_fig_size[0] + legend_margin, n_rows*self.one_fig_size[1] + 1)
			)
			fig.add_subplot(111, frameon=False)
			# hide tick and tick label of the big axes
			plt.tick_params(axis='both', which='both', bottom=False, top=False,
							left=False, right=False, labelcolor='none')
		else:
			fig, ax = plt.subplots(1, 1,
				figsize= (self.one_fig_size[0] + legend_margin, self.one_fig_size[1] + 1)
			)
			axs = np.array(ax)
		return fig, axs

	def plot_exp(self, ax, paths: list, x: str= None, y= "eprew",
			label= ""):
		""" plot data based on given experiment logs, curve will be compute mean for all logs

		@ Args:
			paths: list[str] a list of absolute path
			x, y: str telling which curve you want to plot (assuming lots of curves in a .csv)
		"""
		all_runs = []
		nframes = None
		for path in paths:
			with open(os.path.join(path, self.log_filename), "r") as f:
				try:
					df = pd.read_csv(f)
					all_runs.append(df[y])
					nframes = list(range(len(df[y]))) if x is None else df[x]
				except:
					print("Exception while reading file ", sys.exc_info()[0], path)

		if len(all_runs) == 0: return

		color = self.color_map[label]
		alpha = 1 if len(paths) == 0 else 0.75

		min_length = min([len(run) for run in all_runs])
		all_runs = np.asarray([run[:min_length] for run in all_runs])
		mean_run = np.mean(all_runs, axis= 0)
		nframes = nframes[:min_length]

		if not label in self.marked_labels:
			self.marked_labels.append(label)
			ax.plot(nframes, mean_run, "-", label= label, color= color, alpha= alpha)
		else:
			ax.plot(nframes, mean_run, "-", color= color, alpha= alpha)

		if all_runs.shape[0] > 1:
			err = np.std(all_runs, axis= 0)
			ax.fill_between(
				nframes, mean_run-err, mean_run+err,
				alpha= 0.2, linewidth= 0.0, color= color
			)
	