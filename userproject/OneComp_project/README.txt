
Izhikevich neuron(s) without any connections
=====================================

This is a minimal example, with only one neuron population (with more or less
neurons depending on the command line, but without any synapses). The neurons
are Izhikevich neurons with homogeneous parameters across the neuron population.
This example project contains a helper executable called "generate_run", 
which compiles and executes the model.

To compile it, navigate to genn/userproject/OneComp_project and type:

msbuild ..\userprojects.sln /t:generate_one_comp_runner /p:Configuration=Release

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
--gpu-device: Sets which GPU device to use for the simulation (defaults to -1 which picks automatically)
--num-neurons: Number of neurons to simulate (defaults to 1)

For a first minimal test, using these defaults and recording results with a base name of `test',the system may be used with:

generate_run.exe test

for Windows users, or:

./generate_run test

for Linux, Mac and other UNIX users. 

This would create a set of tonic spiking Izhikevich neurons with no connectivity, 
receiving a constant identical 4 nA input.

Another example of an invocation that runs the simulation using the CPU rather than GPU, 
records timing information and 4 neurons would be: 

generate_run.exe --cpu-only --timing --num_neurons=4 test

for Windows users, or:

./generate_run --cpu-only --timing --num_neurons=4 test

for Linux, Mac and other UNIX users.
