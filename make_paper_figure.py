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

if __name__ == "__main__":
	# auto_plot() # You can used the built-in implementation to plot your curves
	manual_plot()
