# Exptools

A unified experiment deploy, logging, visualizatoin, comparsion tool (based on Tensorboard)

## Related Works

- It seems like [scared](https://github.com/IDSIA/sacred) has already a series of good implementation.
    Although there is no feature in launching, it deserves further check.

- There are good implementations of launching and logging experiment from [rlpyt](https://github.com/astooke/rlpyt).
    But no beautiful viewer to use ([viskit](https://github.com/vitchyr/viskit) is an option, but it is not good enough)

## Installation

- If you want to use it as your project module. In other words, you want to use it as a product.

    ```bash
    cd ${this_project_path}
    python setup.py install
    ```

- If you want to install it and keep on developing the repository. In other words, you want to develop it.

    ```bash
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
    (go to `logging.logger`)
- [x] Auto Prefix for logging title (python context manager should be good)
    (go to `logging.context.logger_context()` and `logging.logger.prefix()`)
- [ ] Customized iteration number when logging 
    It seems no longer needed because they usually log iteration number in the tabular.
- [x] Different types of snapshot and resuming method (go to `logging.save_itr_params`)
- [x] Logging multiple types of data and easy to view (Tensorboard protocol seems good)
    Using viskit philosophy, I fix the comparing issue

### Viewing after or during an experiment

- [x] Beautiful scalar curve (It should be great to export directly for paper/reports)
    Viskit is not beautiful enough, but will do.
- [x] Compare between different variant
    * Automatically extract the difference between each experiment.
    * Require procotol of saving the variant.
    (go to `viewing`, I copied code from [viskit](https://github.com/vitchyr/viskit) and fixing the variant file name problem)
    NOTE: viewing is now in-seperatable from logging.
- [x] Export CSV file for the scalar data
    * It should export every frame, not like Tensorboard who downsampled the curve longer than 1k frames.
    (progress.csv in your log_dir is what you need)

## Usage (API requirement)

You can see demo from `launch_demo.py` and `demo.py`

### Launching

1. Script loading configuration and building variants

    ```python
    from exptools.collections import AttrDict
    from exptools.launching.exp_launcher import run_experiments
    from exptools.launching.variant import VariantLevel, update_config
    from exptools.launching.affinity import encode_affinity, quick_affinity_code

    # Either manually set the resources for the experiment:
    affinity_code = encode_affinity(
        n_cpu_core=4,
        n_gpu=4,
        # hyperthread_offset=8,  # if auto-detect doesn't work, number of CPU cores
        # n_socket=1,  # if auto-detect doesn't work, can force (or force to 1)
        cpu_per_run=1,
        set_affinity=True,  # it can help to restrict workers to individual CPUs
    )
    # Or try an automatic one, but results may vary:
    # affinity_code = quick_affinity_code(n_parallel=None, use_gpu=True)

    # setup your configurations, or you can build via VariantLevel to make cross 
    # combination.
    variant = AttrDict(...)

    run_experiments(
        script= "path/to/your/experiment/script",
        affinity_code= affinity_code,
        experiment_title= "experiment title",
        runs_per_setting= THE_number_of_experiment_you_will_do_in_one_variant,
        variants= [variant],
        log_dirs= [log_dir], # the directory under "${experiment title}"
    )
    ```

    If any types of attribute not found error occurred, that should be missing from your launch file.

2. Script running experiment

    Your actual script that carries out the experiment should be in this following API.

    ```python
    from exptools.launching.variant import load_variant
    from exptools.launching.affinity import affinity_from_code
    import sys, os, json
    import ...

    def main(affinity_code, log_dir, run_id, *args):
        variant = load_variant(log_dir)
        # Then variant is a AttrDict, where you can access your configurations
        # as attribute or dictionary.
        gpu_idx = affinity_from_code(affinity_code)["cuda_idx"]
        # This helps you know what GPU is recommand to you for this experiment
        ...

    if __name__ == "__main__":
        # The main function name has to be "main" or "build_and_train" (conpatible with rlpyt)
        main(*sys.argv[1:]) # the argument will be put as follows:
            # ${affinity_code} ${log_dir} ${run_id} ${*args}
    ```

    Then, you can do whatever you want, even with your own logging mechanism

### Tips

- PyTorch:

    * If you are setting a single GPU for your one experiment, use `torch.cuda.set_device(cuda_idx)`,
    then you can use `device = torch.device("cuda")` to get device during the experiment
