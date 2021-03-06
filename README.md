# OpSNN : Optimized Pipeline for Spiking Neural Networks

## Project info

**Title:**
Optimized Pipeline for the modelling of Spiking Neural Networks (SNNs)

**Project lead and collaborators:**

Lead: Alberto Vergani [@albertovergani](https://github.com/albertovergani)

Lead: Laurent Perrinet [@laurentperrinet](https://github.com/laurentperrinet)

Collaborator: Julia Sprenger [@juliasprenger](https://github.com/juliasprenger)

## Description
We are doing Spiking Neural Networks of the primary visual cortex designed to help us better understand visual computations using Spatio-temporal Diffusion Kernels and Traveling Waves. We are using neural simulators using classical pipelines (pyNN AND (Nest OR SpiNNaker) ), but for which we wish to optimize the different steps: (1) setting up the network, (2) running the
simulation & (3) analyzing the results.

We wish to go beyond the classical strategy ("yet another model") but to understand "why" such a given network would be a good descriptor of neural computations (for a context, see https://arxiv.org/ftp/arxiv/papers/2004/2004.07580.pdf ). With such an efficient simulation pipeline, we would like in the future to "close the loop" and explore the space of all network  configurations, in normal as well as pathological conditions.

Links of interest:

* https://github.com/NeuralEnsemble/PyNN

* https://github.com/nest/nest-simulator

* http://spinnakermanchester.github.io/

* to create an account and run your simulation on a **real** Spinnaker board, sign in @ https://spinn-20.cs.man.ac.uk/hub/login

## Goals for Brainhack Marseille
- working goals: handle the interface between simulations blocks (network building, running simulations, results analysis)
- perspective goal: thinking about closing the loop by optimizing the network structure based on the output of the analysis.

**Skills:**
python 100%
numpy 80%
PyNN 40%

**Striking Image**
![brainhack2020_2](https://user-images.githubusercontent.com/17125783/100328549-ee226f00-2fcc-11eb-84fd-8965dc9a6417.png)


## Wrap-up of results:

We first investigated existing solutions / efforts in that direction :

 * a full simulation pipeline = http://neuralensemble.org/docs/mozaik/index.html (not actively developed now, py3 now supported)
 * NEO = *the* interchange format: https://neo.readthedocs.io/en/latest/index.html for simulations and experiments
 * comparing Nest and spinnaker : these are different simulators on different hardware which are comparable in some aspects (for instance see https://www.frontiersin.org/articles/10.3389/fnins.2018.00291 : "At this setting, NEST and SpiNNaker have a comparable energy consumption per synaptic event."), yet the later being a neuromorphic hardware (for which differential equations governing the dynamics of the neurons are actually implemented in the electronics of the chips, not numerically), it may scale up better in some situations.

### output

we tested different backends for writing files, while keeping neo files (and thus the same plotting functions).

For the testing, we run simulations on SpiNNaker with 1000 cells (https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/output/3_boilerplate.ipynb) obtaining:

* 1 PR : https://github.com/NeuralEnsemble/PyNN/pull/695 (merged! :bowtie: )

#### pkl format
```
Nodes                  : 1
Number of Neurons      : 1000
Excitatory conductance : 4 nS
Inhibitory conductance : 51 nS
Excitatory rate        : 0.84 Hz
Inhibitory rate        : 0.73 Hz
Build time             : 0.00574541 s
Simulation time        : 68.6002 s
Writing time           : 0.246068 s
```

#### nixio format
```
Nodes                  : 1
Number of Neurons      : 1000
Excitatory conductance : 4 nS
Inhibitory conductance : 51 nS
Excitatory rate        : 1.07875 Hz
Inhibitory rate        : 1.095 Hz
Build time             : 0.00556111 s
Simulation time        : 66.1884 s
Writing time           : 212.847 s
```

The writing time regards the saving of spikes for 1000 cells, but voltage for two cells (i.e., [0] and [1])

#### voltage comparison between pkl and nixio format
Overlapped since identical results

![brainhack2020_comparison](https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/output/comparisonEqual.png)

check extended results (spikes and voltage) here https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/output/3_C_loading_inputs.ipynb

#### summary

* in summary, we compared Neo pickle vs Neo-Nix as interchange file formats
  * pickle: faster saving time, smaller file sizes, requires identical environment for reading
  * nix: slower in saving, larger file sizes, interoperable hdf5 file, less dependent on package versions (see also update on https://github.com/NeuralEnsemble/python-neo/issues/310)


### input

we have now the possibility to import tonic datasets into pyNN:

<img src="https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/input/output/test_stop.gif?raw=true" alt="output spikes" class="bg-primary" width="200px"><img src="https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/input/output/input_movie.gif?raw=true" alt="output spikes" class="bg-primary" width="200px"><img src="https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/input/output/output_spike.gif?raw=true" alt="output spikes" class="bg-primary" width="200px"><img src="https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/input/output/nmnist_spike.gif?raw=true" alt="output spikes" class="bg-primary" width="200px">

check out https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/input/D_tonic2SpikeSourceArray.ipynb

* 1 PR : https://github.com/neuromorphs/tonic/pull/89 (merged! :bowtie: )

### benchmark
using

```
  N_pop=1000,  # number of cells
  simtime=1000, # (ms) simulaton duration
```

we get the following results


#### on nest with hdf5:
population size | bio simtime
------ | ------
![2020-12-04_scan_nest__N_pop](https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/benchmark/2020-12-04_scan_nest_N_pop.png) | ![2020-12-04_scan_nest__simtime](https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/benchmark/2020-12-04_scan_nest_simtime.png)

```
buildCPUTime (ms) = 0.583 * N_pop + 0.016/1000 * simtime (ms) * N_pop
simCPUTime (ms) = -0.007 * N_pop + 2.841/1000 * simtime (ms) * N_pop
writeCPUTime (ms) = 0.202 * N_pop + 0.035/1000 * simtime (ms) * N_pop

```


#### on nest with nixio:
```
buildCPUTime (ms) = 0.588 * N_pop + 0.017 * simtime (ms) * N_pop/1000
simCPUTime (ms) = 0.047 * N_pop + 3.139 * simtime (ms) * N_pop/1000
writeCPUTime (ms) = 19.241 * N_pop + 0.134 * simtime (ms) * N_pop/1000
```


#### on spinnaker:

population size | bio simtime
------ | ------
![2020-12-04_scan_spinnaker_N_pop](https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/benchmark/2020-12-04_scan_spinnaker_N_pop.png)  | ![2020-12-04_scan_spinnaker_simtime](https://github.com/SpikeAI/2020-12_brainhack_Project7/blob/main/benchmark/2020-12-04_scan_spinnaker_simtime.png)  

```
buildCPUTime (ms) = 0.002 * N_pop + 0.000/1000 * simtime (ms) * N_pop
simCPUTime (ms) = 24.476 * N_pop + 6.483/1000 * simtime (ms) * N_pop
writeCPUTime (ms) = 0.253 * N_pop + 0.186/1000 * simtime (ms) * N_pop
```
