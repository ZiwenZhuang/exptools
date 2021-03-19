''' The demo telling what is the necessary steps to do, in order
    to reviece parameters and run your experiment.
'''
from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context
from exptools.logging import logger
import sys
import os
import numpy as np

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
    # all gpus you should assign to your system
    # (with respect to all other experiments launched in parallel)
    gpu_idxs = affinity.get("cuda_idx", None)

    print(affinity)
    print(os.environ.get("CUDA_VISIBLE_DEVICES", None))
    print(os.environ.get("CONDA_DEFAULT_ENV", None))

    # under a logger context, run your experiment.
    with logger_context(log_dir, run_ID, name, config):
            # all logging files will be stored under log_dir/run_ID/
        logger.log_text("Start running experiment", 0)
        for epoch_i in range(10):
            # log text whenever you want
            logger.log_text("This is a test", epoch_i, color= "blue")

            # log your scalar with this function
            logger.log_scalar("metric1", epoch_i, epoch_i)
            logger.log_scalar("metric2", config["optionB"]*epoch_i, epoch_i)

            # logging a batch of scalar with logger
            logger.log_scalar_batch("metric_array",
                [config["optionB"]*epoch_i, config["optionC"]*epoch_i],
                epoch_i
            )

            # You can add another scalar file
            additional_csv = "iteration{}.csv".format(epoch_i)
            with logger.additional_scalar_output(additional_csv):
                for itr_i in range(5):
                    logger.log_scalar("inner_step", itr_i, itr_i, filename= additional_csv)
                    logger.log_scalar("inner_metric", itr_i*2 - epoch_i, itr_i, filename= additional_csv)
                    # Don't forget to dump this file
                    logger.dump_scalar(additional_csv)

            # dump all logs into csv file (This is the exact function that
            # write all data into progress.csv file, by default)
            logger.dump_data()

            # You can log images or gifs
            image = (np.random.random((3, 50, 50)) * 256).astype(np.uint8)
            logger.log_image("random_img", image, epoch_i)
            logger.log_gif(
                "random_gif",
                [(np.random.random((3, 32, 32)) * 256).astype(np.uint8) for _ in range(10)],
                epoch_i,
                duration= 0.1
            )

# Or you can also define your main entrance with this function name.
def main(*args):
    build_and_train(*args)
    
if __name__ == "__main__":
    build_and_train(*sys.argv[1:])