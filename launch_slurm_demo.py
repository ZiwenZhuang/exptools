""" A demo of lauch file to run your experiment on slurm.
You don't have to change your experiment entrance script.
"""
from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.slurm import build_slurm_resource
from exptools.launching.exp_launcher import run_on_slurm

default_config = dict(
    optionA = dict(
        choiceA = "hello",
        choiceB = "great",
    ),
    optionB = 2
)

def main(args):
    experiment_title = "demo_experiment"
    slurm_resource = build_slurm_resource(
        mem= "12G",
        n_gpus= 1,
        cuda_module= "cuda-80",
    )
    # NOTE: This resources is specifying for each run of experiment, which is different
    # from making an affinity.
    # For activating conda environment, you should activate when running the lauch script.

    # set up variants
    variant_levels = list()

    values = [
        ["one",],
        ["two",],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("optionA", "choiceB")] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        ["good", int(1e-3)],
        ["better", int(1e3)],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("optionA", "choiceB"), ("optionB",)]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # get all variants and their own log directory
    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)

    run_on_slurm(
        script= "demo.py",
        slurm_resource= slurm_resource,
        experiment_title= experiment_title,
        runs_per_setting= 1,
        variants= variants,
        log_dirs= log_dirs,
        debug_mode= args.debug
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', help= 'A common setting of whether to entering debug mode for remote attach',
        type= int, default= 0,
    )

    args = parser.parse_args()
    # setup for debugging if needed
    if args.debug > 0:
        # configuration for remote attach and debug
        import ptvsd
        import sys
        ip_address = ('0.0.0.0', 5050)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=ip_address, redirect_output= True)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Process attached, start running into experiment...", flush= True)
        ptvsd.break_into_debugger()

    main(args)