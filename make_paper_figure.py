""" a demo script telling you how to make figure for your paper
"""
import numpy as np
from exptools.viewing.plotter import PaperCurvePlotter, color_defaults

def auto_plot():
	paths = ["/root/exptools/data/local/demo_experiment"]
	plotter = PaperCurvePlotter(paths)
	plotter.make_plots(
		args_in_figures= {"optionA.choiceA": None,},
		args_in_series= {"optionB": [2000, 1000, 500]},
		x_key= "metric1",
		y_key= "metric2",
		xlabel= "value1",
		ylabel= "value2",
		x_lim= (-2, 200),
		y_lim= (-10, 200),
		fig_name_shorten_level= 1,
		show_legend= True,
	)

import matplotlib.pyplot as plt
import numpy as np

def manual_plot():
	paths = ["/root/exptools/data/local/demo_experiment"]
	plotter = PaperCurvePlotter(paths)
	curves = plotter.get_curves(
		configs= {
			"optionB": [2000, 1000, 500],
		},
		x_key = "metric1",
		y_key = "metric2",
	)
	fig = plt.figure()
	ax = fig.subplots()
	for i, (config, data) in enumerate(curves):
		# assuming you know each of your data has the same length
		ys = np.stack([d[1] for d in data])
		y_mean = np.nanmean(ys, axis= 0)
		not_nan = ~np.isnan(y_mean)
		x = data[0][0][not_nan]
		y_mean = y_mean[not_nan]
		y_std = np.std(ys[:, not_nan], axis= 0)
		ax.plot(x, y_mean, "-",
			label= "optionB: " + str(config["optionB"]),
			color= color_defaults[i],
			alpha= 0.9
		)
		ax.fill_between(x, y_mean-y_std, y_mean+y_std, linewidth= 0.0, color= color_defaults[i], alpha= 0.2)
	fig.legend()
	plt.show()

from exptools.exp_database import ExpDatabase
from exptools.colors import get_rgb_colors
import os.path as osp
import pandas as pd

def get_data_paths_manually():
	""" This is a demo, but not garantee the successful running. """
	paths = ["/root/exptools/data/local/demo_experiment"]
	database = ExpDatabase(paths)
	datapaths = database.get_datapaths(
		configs= {
			"optionB": [2000, 1000, 500],
		},
	)
	fig = plt.figure()
	ax = fig.subplots()
	color_defaults = get_rgb_colors(len(datapaths))
	for i, (config, data_paths) in enumerate(datapaths):
		ys = []
		for filename in [osp.join(dp, "progress.csv") for dp in data_paths]:
			with open(filename, "r") as f:
				df = pd.read_csv(f)
				ys.append(df["metric2"].values)
			x = df["metric2"]

		# extract and plot the data
		ys = np.stack(ys)
		y_mean = np.nanmean(ys, axis= 0)
		not_nan = ~np.isnan(y_mean)
		x = x[not_nan]
		y_mean = y_mean[not_nan]
		y_std = np.std(ys[:, not_nan], axis= 0)
		ax.plot(x, y_mean, "-",
			label= "optionB: " + str(config["optionB"]),
			color= color_defaults[i],
			alpha= 0.9
		)
		ax.fill_between(x, y_mean-y_std, y_mean+y_std, linewidth= 0.0, color= color_defaults[i], alpha= 0.2)
	fig.legend()
	plt.show()

if __name__ == "__main__":
	# auto_plot() # You can used the built-in implementation to plot your curves
	manual_plot()
	# get_data_paths_manually()
