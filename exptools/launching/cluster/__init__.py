""" a series of script implementation for using cluster managers
"""

class ClusterHandlerBase:

    # customed for each cluster managing utilities
    call_script_cmd = "cluster"
    cluster_manager_name = "cluster"

    # used as affinity signiture for the affinity from code, 
    # NOTE: please don't change this
    affinity_code = "cluster"

    def __init__(self, **kwargs):
        """ Initialize to build resources and ready to make cluster manager
        script.
        """
        pass

    def make_script(self, log_dir, script_name, run_ID, call_command):
        """ build script command that using system call to launch cluster utilities.
        """
        raise NotImplementedError("You should provide specific cluster handler to run_on_cluster function")

