
Pulse-coupled Izhikevich network
================================

This example model is inspired by simple thalamo-cortical network of Izhikevich 
with an excitatory and an inhibitory population of spiking neurons that are
randomly connected. It creates a pulse-coupled network with 80% excitatory 20%
inhibitory connections, each connecting to a fixed number of neurons with sparse connectivity.

To compile it, navigate to genn/userproject/Izh_sparse_project and type:

msbuild ..\userprojects.sln /t:generate_izh_sparse_runner /p:Configuration=Release

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
--num-neurons: Number of neurons (defaults to 10000)
--num-connections: Number of connections per neuron (defaults to 1000)
--gscale: General scaling of synaptic conductances (defaults to 1.0)

An example invocation of generate_run using these defaults and recording results with a base name of `test' would be:

generate_run.exe test

for Windows users, or:

./generate_run test

for Linux, Mac and other UNIX users.

This would create a pulse coupled network of 8000 excitatory 2000 inhibitory
Izhikevich neurons, each making 1000 connections with other neurons, generating
a mixed alpha and gamma regime. For larger input factor, there is more
input current and more irregular activity, for smaller factors less
and less and more sparse activity. The synapses are of a simple pulse-coupling
type. The results of the simulation are saved in the directory `outdir_output`.

Another example of an invocation that runs the simulation using the CPU rather than GPU, 
records timing information and doubles the number of neurons would be: 

generate_run.exe --cpu-only --timing --num_neurons=20000 test

for Windows users, or:

./generate_run --cpu-only --timing --num_neurons=20000 test

for Linux, Mac and other UNIX users.
