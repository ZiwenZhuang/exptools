# Exptools
A unified experiment deploy, logging, visualizatoin, comparsion tool (based on Tensorboard)

# Dead
It seems like [scared](https://github.com/IDSIA/sacred) has already a series of good implementation.
Although there is no feature in launching, it deserves further check.

# Expected features
## Launching experiment
- [ ] Automatic generate variants for all hyperparameters
- [ ] Save variant to a Json file
- [ ] Different method of running experiment in parallel
- [ ] Unified interface for entering an experiment (between this launcher and the experiment)

## Running
- [ ] Ubiquitous logger in every layer of the experiment code (accessing the same object)
- [ ] Auto Prefix for logging title (python context manager should be good)
- [ ] Customized iteration number when logging
- [ ] Different types of snapshot method
- [ ] Logging multi types of data (Tensorboard protocol seems good)

## Visualization
- [ ] Beautiful scalar curve (It should be great to export directly for paper/reports)
- [ ] Compare between different variant
  * Automatically extract the difference between each experiment.
  * Require procotol of saving the variant.
- [ ] Export CSV file for the scalar data
  * It should export every frame, not like Tensorboard who downsampled the curve longer than 1k frames.
