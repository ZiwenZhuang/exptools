''' A demo file telling you how to configure variant and launch experiments
    NOTE: By running this demo, it will create a `data` directory from where you run it.
'''
from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.affinity import encode_affinity, quick_affinity_code
from exptools.launching.exp_launcher import run_experiments

default_config = dict(
    optionA = dict(
        choiceA = "hello",
        choiceB = "great",
    ),
    optionB = 2,
    optionC = 1e5,
)

def main(args):
    experiment_title = "demo_experiment"

    # set up variants
    variant_levels = list()

    values = [
        ["one",],
        ["two",],
        ["three",],
        ["four",],
        ["five",],
        ["six",],
        ["seven",],
        ["eight",],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("optionA", "choiceA")] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        ["good", int(2e3)],
        ["better", int(1e3)],
        ["best", int(5e2)],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("optionA", "choiceB"), ("optionB",)]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # get all variants and their own log directory
    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)

    if args.where == "local":
        from exptools.launching.affinity import encode_affinity, quick_affinity_code
        from exptools.launching.exp_launcher import run_experiments
        # NOTE: you can use quick_affinity_code for simplicity
        affinity_code = encode_affinity(
            n_cpu_core= 12,
            n_gpu= 4,
            contexts_per_gpu= 3,
        )
        run_experiments(
            script= "demo.py",
            affinity_code= affinity_code,
            experiment_title= experiment_title + ("--debug" if args.debug else ""),
            runs_per_setting= 1, # how many times to run repeated experiments
            variants= variants,
            log_dirs= log_dirs,
            debug_mode= args.debug, # if greater than 0, the launcher will run one variant in this process)
        )
    elif args.where == "slurm":
        from exptools.launching.slurm import build_slurm_resource
        from exptools.launching.exp_launcher import run_on_slurm
        slurm_resource = build_slurm_resource(
            mem= "16G",
            time= "3-12:00:00",
            n_gpus= 1,
            partition= "",
            cuda_module= "cuda-10.0",
        )
        run_on_slurm(
            script= "demo.py",
            slurm_resource= slurm_resource,
            experiment_title= experiment_title + ("--debug" if args.debug else ""),
            # experiment_title= "temp_test" + ("--debug" if args.debug else ""),
            script_name= experiment_title,
            runs_per_setting= 1,
            variants= variants,
            log_dirs= log_dirs,
            debug_mode= args.debug, # don't set debug when run on slurm, it is not implemented
        )
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', help= 'A common setting of whether to entering debug mode for remote attach',
        type= int, default= 0,
    )
    parser.add_argument(
        '--where', help= 'slurm or local',
        type= str, default= "local",
        choices= ["slurm", "local"],
    )

    args = parser.parse_args()
    # setup for debugging if needed
    if args.debug > 0:
        # configuration for remote attach and debug
        import ptvsd
        import sys
        ip_address = ('0.0.0.0', 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=ip_address, redirect_output= True)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Process attached, start running into experiment...", flush= True)
        ptvsd.break_into_debugger()

    main(args)