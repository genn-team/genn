
Genetic algorithm for tracking parameters in a HH model cell
============================================================

This example simulates a population of Hodgkin-Huxley neuron models on the GPU and evolves them with a simple 
guided random search (simple GA) to mimic the dynamics of a separate Hodgkin-Huxley
neuron that is simulated on the CPU. The parameters of the CPU simulated "true cell" are drifting 
according to a user-chosen protocol: Either one of the parameters gNa, ENa, gKd, EKd, gleak,
Eleak, Cmem are modified by a sinusoidal addition (voltage parameters) or factor (conductance or capacitance) - 
protocol 0-6. For protocol 7 all 7 parameters undergo a random walk concurrently.

To compile it, navigate to genn/userproject/HHVclampGA_project and type:

msbuild ..\userproject.sln /t:generate_hhvclamp_runner /p:Configuration=Release

for Windows users, or:

make

for Linux, Mac and other UNIX users.


USAGE
-----

generate_run <CPU=0, GPU=1> <protocol> <nPop> <totalT> <outdir> 

Mandatory parameters: 
GPU/CPU: Whether to use the GPU (1) or CPU (0) for the model neuron population
protocol: Which changes to apply during the run to the parameters of the "true cell"
nPop: Number of neurons in the tracking population
totalT: Time in ms how long to run the simulation 
outdir: The directory in which to save results

Optional arguments:
DEBUG=0 or DEBUG=1 (default 0): Whether to run in a debugger
FTYPE=DOUBLE of FTYPE=FLOAT (default FLOAT): What floating point type to use
REUSE=0 or REUSE=1 (default 0): Whether to reuse generated connectivity from an earlier run
CPU_ONLY=0 or CPU_ONLY=1 (default 0): Whether to compile in (CUDA independent) "CPU only" mode.

An example invocation of generate_run is:

generate_run.exe 1 -1 12 200000 test1

for Windows users, or:

./generate_run 1 -1 12 200000 test1

for Linux, Mac and other UNIX users.

This will simulate nPop= 5000 Hodgkin-Huxley neurons on the GPU which will for 1000 ms be matched to a
Hodgkin-Huxley neuron where the parameter gKd is sinusoidally modulated. The output files will be
written into a directory of the name test1_output, which will be created if it does not yet exist.

Another example of an invocation would be: 

generate_run.exe 0 -1 12 200000 test1 FTYPE=DOUBLE CPU_ONLY=1

for Windows users, or:

./generate_run 0 -1 12 200000 test1 FTYPE=DOUBLE CPU_ONLY=1

for Linux, Mac and other UNIX users.
