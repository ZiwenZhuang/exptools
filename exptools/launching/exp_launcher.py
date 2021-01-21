
import subprocess
import datetime
import time
import os
import os.path as osp
import random
import importlib.util

from exptools.launching.affinity import get_n_run_slots, prepend_run_slot, affinity_from_code
from exptools.logging.context import get_log_dir, SLURM_EXP
from exptools.launching.variant import save_variant

from exptools.launching.slurm import SlurmResource, make_sbatch_script


def log_exps_tree(exp_dir, log_dirs, runs_per_setting):
    """ write the experiment process ID and their log_dir names
    into $exp_dir/experiments_tree.txt
    """
    os.makedirs(exp_dir, exist_ok=True)
    with open(osp.join(exp_dir, "experiments_tree.txt"), "a") as f:
        now = datetime.datetime.now()  # dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        timestamp = f"{timestamp} | "
        f.write(f"{timestamp}Experiment manager process ID: {os.getpid()}.\n")
        f.write(f"{timestamp}Number of settings (experiments) to run: "
            f"{len(log_dirs)}  ({runs_per_setting * len(log_dirs)}).\n\n")
        [f.write(timestamp + log_dir + "\n") for log_dir in log_dirs]


def log_num_launched(exp_dir, n, total):
    """ write the total number of experiment launched into $exp_dir/num_launched.txt
    """
    with open(osp.join(exp_dir, "num_launched.txt"), "a") as f:
        now = datetime.datetime.now()  # dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        timestamp = f"{timestamp} | "
        f.write(f"{timestamp}Experiments launched so far: {n} out of {total}.\n")

def make_call_command(script, slot_affinity_code, log_dir, run_ID, args):
    """ A common protocol to make a call string to experiment entrance file """
    call_command = ["python", script, slot_affinity_code, log_dir, str(run_ID)]
    call_command += [str(a) for a in args]
    return call_command

def launch_experiment(script, run_slot, affinity_code, log_dir, variant, run_ID, args, new_process: bool= True):
    """ 
    Parameters
    ----------
        log_dir: the abspath to save variant and feed to running script
        script: the name of experiment script you wish to run
        variant: a dict-like object that tells the experiment configuration
        new_process: a boolean showing whether to deploy a new process to run the experiment
    """
    slot_affinity_code = prepend_run_slot(run_slot, affinity_code)
    affinity = affinity_from_code(slot_affinity_code)
    call_list = list()
    if isinstance(affinity, dict) and affinity.get("all_cpus", False):
        cpus = ",".join(str(c) for c in affinity["all_cpus"])
    elif isinstance(affinity, list) and affinity[0].get("all_cpus", False):
        cpus = ",".join(str(c) for aff in affinity for c in aff["all_cpus"])
    else:
        cpus = ()
    if cpus:
        call_list += ["taskset", "-c", cpus]  # PyTorch obeys better than just psutil.
    call_command = make_call_command(script, slot_affinity_code, log_dir, run_ID, args)
    save_variant(variant, log_dir)
    if new_process:
        print("\ncall string:\n", " ".join(call_list + call_command))
        p = subprocess.Popen(call_list + call_command) # the script recieve from its script name
    else:
        print("\nexperiment function:\n", " ".join(call_command[1:]))
        # load experiment script as module by its path refering to https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        file_abspath = os.path.abspath(script)
        module_name = file_abspath.split("/")[-1][:-3] # exclude ".py" chars
        module_spec = importlib.util.spec_from_file_location(name= module_name, location= file_abspath)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        # call experiment
        if hasattr(module, "main"):
            module.main(*call_command[2:]) # feed the command start from after the script name
        elif hasattr(module, "build_and_train"):
            module.build_and_train(*call_command[2:])
        p = None
    return p


def run_experiments(script, affinity_code, experiment_title, runs_per_setting,
        variants, log_dirs, common_args=None, runs_args=None, debug_mode: int= 0):
    """ For each variant, run experiment whose data will store on your '${ProjectPath}/data/${date}/...'
        NOTE: If provided, 'variants', 'log_dirs' and 'runs_args' should have the same length.
    Parameters
    ----------
        script: a string give the path (start from your project root directory) to the experiment script
        affinity_code: a string specifying Processor Affinity
        variants: a list of AttrDict object that tells each experiment configuration.
        log_dirs: the dir name inside exp_dir, it has to have the same length as variants.
        debug_mode: 0 for not debuging; 1 for debuging by running experiment one-by-one; 2 for debuging by sampling one from variants.
    """
    n_run_slots = get_n_run_slots(affinity_code)
    exp_dir = get_log_dir(experiment_title)
    procs = [None] * n_run_slots
    common_args = () if common_args is None else common_args
    assert len(variants) == len(log_dirs)
    if runs_args is None:
        runs_args = [()] * len(variants)
    assert len(runs_args) == len(variants)
    log_exps_tree(exp_dir, log_dirs, runs_per_setting)
    num_launched, total = 0, runs_per_setting * len(variants)
    for run_ID in range(runs_per_setting):
        if debug_mode > 0:
            # special case, run only in this process to enable debug
            num_variants = len(variants)
            runs_order = list(range(num_variants)) if debug_mode == 1 else random.sample(list(range(num_variants)), num_variants)
            for i, run_index in enumerate(runs_order):
                variant, log_dir, run_args = variants[run_index], log_dirs[run_index], runs_args[run_index]
                log_dir = osp.join(exp_dir, log_dir)
                os.makedirs(log_dir, exist_ok=True)
                # Now abandom launch_experiment return
                launch_experiment(
                    script= script,
                    run_slot= i,
                    affinity_code= affinity_code,
                    log_dir= log_dir,
                    variant= variant,
                    run_ID= run_ID,
                    args= common_args + run_args,
                    new_process= False # NOTE: This stops launch_experiment from spawn a new process
                )
        else:
            # common case, run with multiprocessing
            for variant, log_dir, run_args in zip(variants, log_dirs, runs_args):
                launched = False
                log_dir = osp.join(exp_dir, log_dir)
                os.makedirs(log_dir, exist_ok=True)
                while not launched:
                    # iterate through 'procs' to find a slot to launch one experiment.
                    for run_slot, p in enumerate(procs):
                        if p is None or p.poll() is not None:
                            procs[run_slot] = launch_experiment(
                                script=script,
                                run_slot=run_slot,
                                affinity_code=affinity_code,
                                log_dir=log_dir,
                                variant=variant,
                                run_ID=run_ID,
                                args=common_args + run_args,
                            )
                            launched = True
                            num_launched += 1
                            log_num_launched(exp_dir, num_launched, total)
                            break
                    if not launched:
                        time.sleep(10)
    for p in procs:
        if p is not None:
            p.wait()  # Don't return until they are all done.

def run_on_slurm(script: str, slurm_resource: SlurmResource, experiment_title: str,
        runs_per_setting: int, variants, log_dirs,
        common_args= None, runs_args= None, debug_mode= 0
    ):
    """ A interface connecting PySbatch and this exptools. All stdout of the experiment will be
    direct to log_dirs (the parent of 'run_ID') named as 'run_ID.stdout'
    @ Args:
        exclude: a string specifying the nodes you want to exclude, might be different depend on
            clusters.
        debug_mode: 0 - no debugging; NOTE: did not support later debugging
    """
    exp_dir = get_log_dir(experiment_title, exp_machine= SLURM_EXP)
    assert len(variants) == len(log_dirs)
    if runs_args is None:
        runs_args = [()] * len(variants)
    common_args = () if common_args is None else common_args
    assert len(runs_args) == len(variants)
    log_exps_tree(exp_dir, log_dirs, runs_per_setting)
    
    # start deploying experiments
    for run_ID in range(runs_per_setting):
        if debug_mode > 0:
            raise NotImplementedError
        else:
            # common case, deploy experiments as slurm jobs one by one
            for variant, log_dir, run_args in zip(variants, log_dirs, runs_args):
                log_dir = osp.join(exp_dir, log_dir)
                os.makedirs(log_dir, exist_ok=True)
                save_variant(variant, log_dir)
                call_command = make_call_command(script, "slurm", log_dir, run_ID, common_args + run_args)
                slurm_script = make_sbatch_script(
                    log_dir= log_dir,
                    experiment_title= experiment_title,
                    run_ID= run_ID,
                    call_command= call_command,
                    slurm_resource= slurm_resource,
                )
                # TODO: acquiree job id after calling sbatch
                os.system("sbatch " + slurm_script)
                print("sbatch deploy on command: \n\t" + " ".join(call_command))
