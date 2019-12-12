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

## Expected features (and where to find it if it is implemented)

### Launching experiments

- [x] Automatic generate variants for all hyperparameters (goto `launching.variant.make_variants`)
- [x] Save variant to a Json file (goto `launching.variant.save_variant`)
- [ ] Different method of running experiment in parallel
- [ ] Unified interface for entering an experiment (between this launcher and the experiment)

### Logging during an experiment

- [ ] Ubiquitous logger in every layer of the experiment code (accessing the same object)
- [ ] Auto Prefix for logging title (python context manager should be good)
- [ ] Customized iteration number when logging
- [ ] Different types of snapshot and resuming method
- [ ] Logging multiple types of data and easy to view (Tensorboard protocol seems good)

### Viewing after an experiment

- [ ] Beautiful scalar curve (It should be great to export directly for paper/reports)
- [ ] Compare between different variant
    * Automatically extract the difference between each experiment.
    * Require procotol of saving the variant.
- [ ] Export CSV file for the scalar data
    * It should export every frame, not like Tensorboard who downsampled the curve longer than 1k frames.
