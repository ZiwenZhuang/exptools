''' The demo telling what is the necessary steps to do, in order
    to reviece parameters and run your experiment.
'''
from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context
import exptools.logging.logger as logger
import sys
import os

# You have to name your main entrance using this name, or you might
# not be able to debug your experiment.
def build_and_train(affinity_code, log_dir, run_ID, **kwargs):
    # I prefer put all tunable default configs into launch file

    # acquire affinity asigned by the launcher.
    # NOTE: If the affinity is a list, it means multiple resources (gpu) 
    # is assigned to current experiment
    affinity = affinity_from_code(affinity_code)

    
    # now you will have `config` as a dictionary in the same
    # structure you define your default configurations
    config = load_variant(log_dir)

    name = "demo_experiment"
    # This helps you know what GPU is recommand to you for this experiment
    gpu_idx = affinity["cuda_idx"]

    print(affinity)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(os.environ["CONDA_DEFAULT_ENV"])

    # under a logger context, run your experiment.
    with logger_context(log_dir, run_ID, name, config):
        logger.log("Start running experiment")
        for epoch_i in range(10):
            # log your scalar with this function for example
            logger.record_tabular("metric1", epoch_i, epoch_i)
            logger.record_tabular("metric2", config["optionB"]*epoch_i, epoch_i)

            # logging an array with logger
            logger.record_tabular_misc_stat("metric_array",
                [config["optionB"]*epoch_i, config["optionC"]*epoch_i],
                epoch_i
            )

            # dump all logs into csv file (This is the exact function that
            # write one line into progress.csv file)
            logger.dump_tabular()

# Or you can also define your main entrance with this function name.
def main(*args):
    build_and_train(*args)
    
if __name__ == "__main__":
    build_and_train(*sys.argv[1:])