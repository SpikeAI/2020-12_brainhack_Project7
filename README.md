# OpSNN : Optimized Pipeline for Spiking Neural Networks

## Project info

**Title:**
Optimized Pipeline for the modelling of Spiking Neural Networks (SNNs)

**Project lead and collaborators:**

Lead: Alberto Vergani @albertovergani

Lead: Laurent Perrinet @laurentperrinet

Collaborator: Julia Sprenger @juliasprenger

**Description:**
We are doing Spiking Neural Networks of the primary visual cortex designed to help us better understand visual computations using Spatio-temporal Diffusion Kernels and Traveling Waves. We are using neural simulators using classical pipelines (pyNN AND (Nest OR SpiNNaker) ), but for which we wish to optimize the different steps: (1) setting up the network, (2) running the
simulation & (3) analyzing the results.
 
We wish to go beyond the classical strategy ("yet another model") but to understand "why" such a given network would be a good descriptor of neural computations (for a context, see https://arxiv.org/ftp/arxiv/papers/2004/2004.07580.pdf ). With such an efficient simulation pipeline, we would like in the future to "close the loop" and explore the space of all network  configurations, in normal as well as pathological conditions.
 
https://github.com/NeuralEnsemble/PyNN

https://github.com/nest/nest-simulator

http://spinnakermanchester.github.io/

**Goals for Brainhack Marseille**
- working goals: handle the interface between simulations blocks (network building, running simulations, results analysis) 
- perspective goal: thinking about closing the loop by optimizing the network structure based on the output of the analysis.

**Skills:**
python 100%
numpy 80%
PyNN 40%

**Striking Image**
![brainhack2020_2](https://user-images.githubusercontent.com/17125783/100328549-ee226f00-2fcc-11eb-84fd-8965dc9a6417.png)
