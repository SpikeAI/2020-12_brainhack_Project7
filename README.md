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


## Wrap-up:

We investigated different existing solutions / efforts in that direction :

 * a full simulation pipeline = http://neuralensemble.org/docs/mozaik/index.html (not actively developped now, py3 now supported)
 * NEO = *the* interchange format: https://neo.readthedocs.io/en/latest/index.html for simulations and experiments


### output

we tested different backends for writing files, while keeping neo files (and thus the same plotting functions).

--- Vogels-Abbott Network Simulation ---

neo==0.9.0 | nest | nixio | 400 neurons

Nodes                  : 1
Number of Neurons      : 400
Number of Synapses     : 1320 e→e,i  372 i→e,i
Excitatory conductance : 4 nS
Inhibitory conductance : 51 nS
Excitatory rate        : 0.134375 Hz
Inhibitory rate        : 0.15 Hz
Build time             : 2.72644 s
Simulation time        : 1.84852 s
Writing time           : 12.3382 s

neo==0.9.0 | nest | pkl | 400 neurons

Nodes                  : 1
Number of Neurons      : 4000
Number of Synapses     : 128066 e→e,i  32011 i→e,i
Excitatory conductance : 4 nS
Inhibitory conductance : 51 nS
Excitatory rate        : 23.1569 Hz
Inhibitory rate        : 21.765 Hz
Build time             : 9.64068 s
Simulation time        : 11.8327 s
Writing time           : 0.97302 s

soon also picture...

### input

we have now the possibility to import tonic datasets into pyNN:

<img src="https://github.com/SpikeAI/2020-11_brainhack_Project7/blob/main/input/output/nmnist_spike.gif?raw=true" alt="output spikes" class="bg-primary" width="200px">

check out https://github.com/SpikeAI/2020-11_brainhack_Project7/blob/main/input/D_tonic2SpikeSourceArray.ipynb


### benchmark
using

  N_pop=1000,  # number of cells
  simtime=1000, # (ms) simulaton duration

we get the following results on nest:

![2020-12-04_scan_nest__N_pop](https://github.com/SpikeAI/2020-11_brainhack_Project7/blob/main/benchmark/2020-12-04_scan_nest_N_pop)
![2020-12-04_scan_nest__simtime](https://github.com/SpikeAI/2020-11_brainhack_Project7/blob/main/benchmark/2020-12-04_scan_nest__simtime.png)
