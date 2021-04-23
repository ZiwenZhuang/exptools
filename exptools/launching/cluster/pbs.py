""" Created by Ziwen Zhuang
Automatically writing PBS (Portable Batch System) resources and PBS script and submit each experiment
to the PBD server.
"""
from exptools.launching.cluster import ClusterHandlerBase
import os
from os import path
from collections import namedtuple

qsub_template = """#!/bin/sh
#PBS -l nodes=1
#PBS -l mem={mem}
#PBS -l walltime={walltime}
#PBS -N {job_name}
#PBS -o {stdout}
#PBS -e {stderr}
"""

class PbsHandler(ClusterHandlerBase):

    call_script_cmd = "qsub"
    cluster_manager_name = "pbs"

    def __init__(self,
            mem: str= "32G",
            walltime: str= "7200:00:00",
            n_gpus: int= 0,
            destination_queue: str= None,
            cmd_prefix: str= None,
        ):
        self.mem = mem
        self.walltime = walltime
        self.n_gpus = n_gpus
        self.destination_queue = destination_queue
        self.cmd_prefix = cmd_prefix

    def make_script(self, log_dir, script_name, run_ID, call_command):
        """ build qsub script for pbs according to configuration
        """
        qsub_string = qsub_template.format(
            mem= self.mem,
            walltime= self.walltime,
            job_name= script_name,
            stdout= path.join(log_dir, "run_{}.stdout".format(run_ID)),
            stderr= path.join(log_dir, "run_{}.stderr".format(run_ID)),
        )
        if not self.destination_queue is None:
            qsub_string += "\n#PBS -q {}".format(self.destination_queue)
        if self.n_gpus > 0:
            qsub_string += "\n#PBS -W x=GRES:gpu@{}".format(str(self.n_gpus))
        
        ###### Load a `cmd_prefix` for the flexibility of PBS usage ######
        if self.cmd_prefix is not None:
            qsub_string += "\n"
            qsub_string += self.cmd_prefix
            qsub_string += "\n"

        # now is time to add call_command and the last piece of the script
        qsub_string += "\n"
        qsub_string += " ".join(call_command)

        with open(path.join(log_dir, "{}_{}.qsub".format(script_name, run_ID)), "w") as f:
            f.write(qsub_string)
        return path.join(log_dir, "{}_{}.qsub".format(script_name, run_ID))
