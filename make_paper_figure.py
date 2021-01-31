""" a demo script telling you how to make figure for your paper
"""
from exptools.viewing.plotter import PaperCurvePlotter

def main():
	paths = ["/root/exptools/data/local/demo_experiment"]
	plotter = PaperCurvePlotter(paths,
		x_lim= (-2, 200),
		y_lim= (-10, 200),
		fig_name_shorten_level= 1,
		show_legend= True,
	)
	plotter.make_plots(
		args_in_figures= {"optionA.choiceA": None,},
		args_in_series= {"optionB": [2000, 1000, 500]},
		x_key= "metric1",
		y_key= "metric2",
		xlabel= "value1",
		ylabel= "value2",
	)

if __name__ == "__main__":
	main()
