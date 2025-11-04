# CellTypeMeanField

This repository contains code and data for the study "Inhibitory cell type heterogeneity in a spatially structured mean-field model of V1" by Kim and Choi (https://doi.org/10.1101/2025.03.13.643046).

### Fig. 1e: Rastor plot under Gaussian input

First compile the simulation code in Matlab using
```mex LIF4pop.c```.

Then run the simulation with
```runSimSpikes('param_base', 300000, 42)```.

Use ```Fig1e.ipynb``` to plot raster.

### Fig. 2: Mean-field model validation

To find mean-field model fixed points with network size 3000000, run ```python param_base 300000```.

To obtain firing rates quickly (without saving spikes), use ```runSimFR('param_base', N, seed)```.

Follow ```Fig2.ipynb``` to compare mean-field model with spiking simulations and analytical solutions. Need to run for multiple sizes before Fig2b-c can be plotted.

### Fig. 3: Stability analysis

Run ```Fig3.ipynb``` for bifurcation/stability analysis.

### Fig. 4: Spatiotemporal dynamics

To run simulations, first create a batch of parameter files using ```python create_batch.py param_base```. 
Then run ```runSimSpike`` for all sigma_e values.
Run ```Fig4.ipynb``` for analysis of spatiotemporal features from spiking.

### Fig. 5: Gain modulation and linear response theory

To compute the mean-field solutions under stimulation conditions, run

```python gain_fp_current.py gain_current control```

for control condition (no inhibitory neuron modulation) and

``` python gain_fp_current.py gain_current pv_pos 0.5 2.0```

for modulatory simulations. In the above example, PV is stimulated at position 0.5 with strength factor 2.0.
The visual input is given at 0.5, so this simulates proximal stimulation. Use 0.0 for distal.
The strength factor is modulates the stimulus to the modulating inhibitory neuron. The strength factor multiplied by the spontaneous input current is the raw stimulation strength.

For conductance-based model, use 

``` python gain_fp_cond.py gain_cond pv_pos 0.5 2.0```

and so on.

Use ```Fig5.ipynb``` for plots and linear response theory analysis.
