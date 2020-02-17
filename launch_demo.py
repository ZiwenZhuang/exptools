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
    optionB = 2
)

def main(args):
    experiment_title = "pearl_reproduction"
    affinity_code = quick_affinity_code(n_parallel=8)
    # NOTE: you can also use encode_affinity to specifying how to distribute each
    # experiment in your computing nodes.

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

    run_experiments(
        script= "demo.py",
        affinity_code= affinity_code,
        experiment_title= experiment_title,
        runs_per_setting= 1, # how many times to run repeated experiments
        variants= variants,
        log_dirs= log_dirs,
        debug_mode= args.debug, # if greater than 0, the launcher will run one variant in this process)
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