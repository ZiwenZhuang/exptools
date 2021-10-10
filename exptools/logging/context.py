
from contextlib import contextmanager
import datetime
import os
import os.path as osp
import json
from copy import deepcopy
import csv

from exptools.logging import logger, Logger
from exptools.logging.console import colorize
from exptools.launching.variant import VARIANT, flatten_variant4hparams, save_variant

# NOTE: you have to run your python command at your project root directory \
# (the parent directory of your 'data' directory)
LOG_DIR = osp.abspath(osp.join(os.getcwd(), 'data'))
SCALAR_LOG_FILE = "progress.csv"
TEXT_LOG_FILE = "debug.log"
PARAMS_LOG_FILE = VARIANT

LOCAL_EXP = "local"

def get_log_dir(experiment_name, exp_machine= LOCAL_EXP):
    """ return string of "${ProjectPATH}/data/{exp_machine}/${date}/{experiment_name}/"
    """
    yyyymmdd = datetime.datetime.today().strftime("%Y%m%d")
    log_dir = osp.join(LOG_DIR, exp_machine, experiment_name, yyyymmdd)
    return log_dir

@contextmanager
def logger_context(log_dir, run_ID, name, log_params=None, snapshot_mode="none", itr_i= 0, refresh= False):
    """ setup the context for one experiment with these parameters.
        And save experiment parameters through 'log_params' as you need. \\
        NOTE: This will look for `data` folder under the directory you run python.

        snapshot_mode: choose between "all", "last", "none", or a int specifying the gap 
    """
    log_dir = osp.join(log_dir, f"run_{run_ID}")
    exp_dir = osp.abspath(log_dir)
    if LOG_DIR != osp.commonpath([exp_dir, LOG_DIR]):
        print(colorize(
            f"logger_context received log_dir outside of {LOG_DIR}: " + \
            f"prepending by {LOG_DIR}/local/<experiment_name>/<yyyymmdd>/",
            color= "yellow"
        ))
        exp_dir = get_log_dir(log_dir)

    logger.set_client(Logger(exp_dir, refresh= refresh))

    logger.add_text_output(TEXT_LOG_FILE)
    logger.add_scalar_output(SCALAR_LOG_FILE)
    logger.push_text_prefix(f"{name}_{run_ID} ")
    print(colorize("Program started, working on...", color= "green"))

    # save params into this experiment directory
    if log_params is None:
        log_params = dict()
    log_params["name"] = name
    log_params["run_ID"] = run_ID
    save_variant(log_params, exp_dir)
    if logger.tb_writer is not None:
        flat_params = flatten_variant4hparams(deepcopy(log_params))
        logger.tb_writer.add_hparams(
            dict(**flat_params),
            dict(z_dummy_metric= 0.0),
            name= "./",
        )
    

    yield

    print(colorize("Program finished, warpping up...", color= "green"))
    logger.pop_text_prefix()
    logger.remove_scalar_output(SCALAR_LOG_FILE)
    logger.remove_text_output(TEXT_LOG_FILE)

    logger.unset_client()
