# Exptools
<img align="right" width="180" height="180" src="figures/logo.png">

A unified experiment deploy, logging, visualizatoin, comparsion tool (based on Tensorboard)


## Related Works

- It seems like [scared](https://github.com/IDSIA/sacred) has already a series of good implementation.
    Although there is no feature in launching, it deserves further check.

- There are good implementations of launching and logging experiment from [rlpyt](https://github.com/astooke/rlpyt).
    But no beautiful viewer to use ([viskit](https://github.com/vitchyr/viskit) is an option, but it is not good enough)

## Installation

- If you want to use it as your project module. In other words, you want to use it as a product.

    ```bash
    git clone https://github.com/ziwenzhuang/exptools
    cd ${this_project_path}
    python setup.py install
    ```
    or
    ```bash
    pip install git+git://github.com/ziwenzhuang/exptools
    ```

- If you want to install it and keep on developing the repository. In other words, you want to develop it.

    ```bash
    git clone https://github.com/ziwenzhuang/exptools
    cd ${this_project_path}
    pip install -e .
    ```

- You can also install with tensorflow/tensorboard installed in your python enrironment. And you can utilize tensorboard feature.

    **Important:** You have to manually install `tensorflow` (no matter if it is GPU version) to make the package compatible for tensorboard.
    
    For futher information, please see `setup.py`

## Expected features (and where to find it if it is implemented)

### Launching experiments

- [x] Automatic generate variants for all hyperparameters (goto `launching.variant.make_variants`)
- [x] Save variant to a Json file (goto `launching.variant.save_variant`)
- [x] Different method of running experiment in parallel (goto `launching.exp_launcher.run_experiments`)
    * You can debug via debug option in `run_experiments` argument. And the function will select a experiment to run at random.
- [x] Unified interface for entering an experiment (between this launcher and the experiment) (goto `launching.exp_launcher.run_experiments`)

### Logging during an experiment

- [x] Ubiquitous logger in every layer of the experiment code (accessing the same object)
    (type `from exptools.logging import logger` and you get it)
    And check `exptools/logging/_logger.py` for API reference
- [x] Auto Prefix for logging title (python context manager should be good)
    (goto `logging.context.logger_context()`)
- [x] Customized iteration number when logging 
    It seems no longer needed because they usually log iteration number in the tabular.
- [ ] Different types of snapshot and resuming method
- [x] Logging multiple types of data and easy to view (Tensorboard protocol seems good)
    (goto `exptools/logging/_logger.py` for reference)
    Some of the data has to be viewed by combining sftp and vscode

### Viewing after or during an experiment

- [x] Beautiful scalar curve online (It should be great to export directly for paper/reports)
    Viskit is not beautiful enough, but will do.
- [x] Compare between different variant
    * Automatically extract the difference between each experiment.
    * Require procotol of saving the variant.
    (go to `viewing`, I copied code from [viskit](https://github.com/vitchyr/viskit) and fixing the variant file name problem)
    NOTE: viewing is now in-seperatable from logging.
- [x] Export CSV file for the scalar data
    * It should export every frame, not like Tensorboard who downsampled the curve longer than 1k frames.
    (progress.csv in your log_dir is what you need)

- [x] Easy to make plots for publishing paper
    * See `make_paper_figure.py`

## Usage (API requirement)

You can see demo from `launch_demo.py` and `demo.py`

### Launching (and script template)

1. Script loading configuration and building variants

    * see `launch_demo.py`

    If any types of attribute not found error occurred, that should be missing from your launch file.

    ```python
    from exptools.launching.variant import VariantLevel, make_variants, update_config
    from exptools.launching.exp_launcher import run_experiments, run_on_cluster

    default_config = dict(
    )

    def main(args):
        experiment_title = "demo_experiment"
        variant_levels = list()

        # values = [
        #     [,],
        # ]
        # dir_names = ["{}".format(*v) for v in values]
        # keys = [(,),]
        # variant_levels.append(VariantLevel(keys, values, dir_names))

        # get all variants and their own log directory
        variants, log_dirs = make_variants(*variant_levels)
        for i, variant in enumerate(variants):
            variants[i] = update_config(default_config, variant)

        if args.where == "local":
            from exptools.launching.affinity import encode_affinity, quick_affinity_code
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
        else:
            if args.where == "slurm":
                from exptools.launching.cluster.slurm import SlurmHandler
                cluster_handler = SlurmHandler(
                    mem= "16G",
                    time= "3-12:00:00",
                    n_gpus= 1,
                    partition= "",
                    cuda_module= "cuda-10.0",
                )
            elif args.where == "pbs":
                from exptools.launching.cluster.pbs import PbsHandler
                cluster_handler = PbsHandler(
                    mem= "16G",
                    walltime= "7200:00:00",
                    n_gpus= 1,
                    destination_queue= "gpu1",
                    cmd_prefix= None,
                )
            run_on_cluster(
                script= "demo.py",
                cluster_handler= cluster_handler,
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
            choices= ["pbs", "slurm", "local"],
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
    ```

2. Script running experiment

    * see `demo.py`

    Your actual script that carries out the experiment should be in this following API.

    ```python
    from exptools.launching.affinity import affinity_from_code
    from exptools.launching.variant import load_variant
    from exptools.logging.context import logger_context
    from exptools.logging import logger

    # You have to name your main entrance using this name, or you might
    # not be able to debug your experiment.
    def build_and_train(affinity_code, log_dir, run_ID, **kwargs):
        affinity = affinity_from_code(affinity_code)

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

    def main(*args):
        build_and_train(*args)
    if __name__ == "__main__":
        build_and_train(*sys.argv[1:])
    ```

    Then, you can do whatever you want, even with your own logging mechanism

### Tips

- PyTorch:

    * If you are setting a single GPU for your one experiment, use `torch.cuda.set_device(cuda_idx)`,
    then you can use `device = torch.device("cuda")` to get device during the experiment
