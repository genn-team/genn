The Cell-Type Specific Cortical Microcircuit
============================================

This example model is a reimplementation of the microcircuit model developed by Tobias C. Potjans and Markus Diesmann.
It is a full-scale spiking network model of the local cortical microcircuit. 
The simulated spontaneous activity is asynchronous irregular and cell-type specific firing rates are in agreement 
with in vivo recordings in awake animals, including the low rate of layer 2/3 excitatory cells.
Potjans, T. C., & Diesmann, M. (2014). The Cell-Type Specific Cortical Microcircuit: Relating Structure and Activity in a Full-Scale Spiking Network Model. Cerebral Cortex, 24(3), 785–806. https://doi.org/10.1093/cercor/bhs358

To compile it, navigate to genn/userproject/PotjansMicrocircuit_project and type:

msbuild ..\userprojects.sln /t:generate_potjans_microcircuit_runner /p:Configuration=Release

for Windows users, or:

make

for Linux, Mac and other UNIX users.


USAGE
-----

generate_run [OPTIONS] <outname> 

Mandatory arguments:
outname: The base name of the output location and output files

Optional arguments:
--debug: Builds a debug version of the simulation and attaches the debugger
--cpu-only: Uses CPU rather than CUDA backend for GeNN
--timing: Uses GeNN's timing mechanism to measure performance and displays it at the end of the simulation
--ftype: Sets the floating point precision of the model to either float or double (defaults to float)
--neuron-scale: Scaling factor for number of neurons (defaults to 0.5)
--connectivity-scale: Scaling factor for connectivity (defaults to 0.5)
--duration: Duration of simulation [ms] (defaults to 1000ms)

An example invocation of generate_run using these defaults and recording results with a base name of `test' would be:

generate_run.exe test

for Windows users, or:

./generate_run test

for Linux, Mac and other UNIX users.

This would create a microcircuit model with 38582 neurons and 74715499 synapses
The results of the simulation are saved in the directory `outdir_output`.

Another example of an invocation that runs the simulation at full-scale for 10s and records timing information would be: 

generate_run.exe --timing --neuron-scale 1.0 --connectivity-scale 1.0 --duration 10000 test

for Windows users, or:

./generate_run --timing --neuron-scale 1.0 --connectivity-scale 1.0 --duration 10000 test

for Linux, Mac and other UNIX users.
