""" Created by Ziwen Zhuang, the solution is refering to Xingyu's solution
A resources-specifying interface for using slurm
"""
from exptools.launching.cluster import ClusterHandlerBase
from exptools.logging.console import colorize
import os
from os import path
from collections import namedtuple

sbatch_template = """#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH -o {stdout}
#SBATCH -e {stderr}
"""

class SlurmHandler(ClusterHandlerBase):

    call_script_cmd = "sbatch"
    cluster_manager_name = "slurm"

    def __init__(self,
            mem: str= "12G",
            time: str= "24:00:00",
            n_gpus: int= 0,
            partition: str= None,
            cuda_module: str= None,
            singularity_img: str= None,
            exclude: str= None,
            cmd_prefix: str= None,
            other_modules: list= None, # a sequence of str
        ):
        """
        Specifying slurm resources refering to https://slurm.schedmd.com/sbatch.html
        @ Args:
            cuda_module: a string telling what cuda module is on your cluster, it has to be provided
                if n_gpus > 0
            partition: "--partition" a string for slurm partition.
            singularity_img: a string telling the absolute path of your singularity image, if not
                provided, no singularity module will be used
            exclude: a string specifying the nodes you want to exclude, might be different depend on
                clusters.
            time: a string with "hh:mm:ss" format telling the running time limit, or "d-hh:mm:ss" for 
                longer time limit.
            cmd_prefix: a string which gives you the flexibility to load what you want. It will be put
                between `#SBATCH` command and `module load` command.
            other_modules: a list of string, which will only be added as `module load ...` into script
        """
        self.mem = mem
        self.time = time
        self.n_gpus = n_gpus
        self.partition = partition
        self.cuda_module = cuda_module
        self.singularity_img = singularity_img
        self.exclude = exclude
        self.cmd_prefix = cmd_prefix
        self.other_modules = other_modules

    def make_script(self,
            log_dir,
            script_name,
            run_ID,
            call_command,
        ):
        """ based on resources specification (for each run), generae one slurm script and write script
        into {log_dir}/run_exp.slurm for you to call. The slurm script will be composed as a series of 
        `#SBATCH` command and a series `module load` command, then the python command at last.
            The default value for resources referring to SlurmResource
        Returns:
            scriptname: the file name of the script to run this experiment. This is absolute path
                if and only if log_dir is an absolute path.
        """
        ###### Adding SBATCH options for requesting resources ######
        sbatch_string = sbatch_template.format(
            mem= self.mem,
            time= self.time,
            stdout= path.join(log_dir, "run_{}.stdout".format(run_ID)),
            stderr= path.join(log_dir, "run_{}.stderr".format(run_ID)),
        )
        if not self.partition is None:
            sbatch_string += "\n#SBATCH --partition={}".format(self.partition)
        if not self.exclude is None:
            sbatch_string += "\n#SBATCH --exclude={}".format(self.exclude)
        if self.n_gpus > 0:
            if self.cuda_module is None:
                print(colorize(
                    "Warning: You want to use GPU but did not provide cuda module",
                    "yellow",
                ))
            sbatch_string += "\n#SBATCH --gres=gpu:{}".format(self.n_gpus)

        ###### Load a `cmd_prefix` for the flexibility of Slurm usage ######
        if self.cmd_prefix is not None:
            sbatch_string += "\n"
            sbatch_string += self.cmd_prefix
            sbatch_string += "\n"
        
        ###### Done requesting resources, start loading module strings ######
        if self.cuda_module is not None:
            sbatch_string += "\nmodule load {}".format(self.cuda_module)

        if self.other_modules is not None:
            for mod in self.other_modules:
                sbatch_string += "\nmodule load " + mod

        if self.singularity_img is not None:
            sbatch_string += "\nmodule load singularity"
            sbatch_string += "\nsingularity exec{nv} {img} ".format(
                nv= " --nv" if self.n_gpus > 0 else "",
                img= self.singularity_img,
            )    

        # now is time to add call_command as the last piece of the script
        sbatch_string += "\n"
        sbatch_string += " ".join(call_command)

        with open(path.join(log_dir, "{}_{}.slurm".format(script_name, run_ID)), 'w') as f:
            f.write(sbatch_string)
        return path.join(log_dir, "{}_{}.slurm".format(script_name, run_ID))


"""
Trying to make debug feature on slurm, but failed. These code can still be reused.
"""
def slurm_debug_command(script, debug_host, log_dir, run_ID, args):
    """ A common protocol to make a call string to slurm debug entrance file
    This command should lead to calling this script directly.
    """
    call_command = ["python", __file__, debug_host, script, log_dir, str(run_ID)]
    call_command += [str(a) for a in args]
    return call_command

def debug_experiment(debug_host:str, script, *exp_args):
    import ptvsd
    import sys
    import importlib.util
    ip_address = debug_host.split(":")
    ip_address[1] = int(ip_address[1])
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % tuple(ip_address), flush= True)
    # Allow other computers to attach to ptvsd at this IP address and port.
    ptvsd.enable_attach(address=ip_address)
    # Pause the program until a remote debugger is attached
    ptvsd.wait_for_attach()
    print("Process attached, start running into experiment...", flush= True)
    ptvsd.break_into_debugger()

    # load script and call as the protocol described
    file_abspath = os.path.abspath(script)
    module_name = file_abspath.split("/")[-1][:-3] # exclude ".py" chars
    module_spec = importlib.util.spec_from_file_location(name= module_name, location= file_abspath)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # call experiment
    if hasattr(module, "main"):
        module.main(*exp_args) # feed the command start from after the script name
    elif hasattr(module, "build_and_train"):
        module.build_and_train(*exp_args)

if __name__ == "__man__":
    """ NOTE: This script should only called by exp_launcher using slurm script.
    """
    import sys
    debug_experiment(*sys.argv[1:])
