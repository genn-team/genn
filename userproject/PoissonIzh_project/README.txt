
Izhikevich network receiving Poisson input spike trains
=======================================================

In this example project there is again a pool of non-connected Izhikevich model neurons
that are connected to a pool of Poisson input neurons with a fixed probability.
This example project contains a helper executable called "generate_run", which compiles and
executes the model.

To compile it, navigate to genn/userproject/PoissonIzh_project and type:

msbuild ..\userprojects.sln /t:generate_poisson_izh_runner /p:Configuration=Release

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
--num-poisson:  Number of Poisson sources to simulate (defaults to 100)
--num-izh: Number of Izhikievich neurons to simulate (defaults to 10)
--pconn: Probability of connection between each pair of poisson sources and neurons (defaults to 0.5)
--gscale: Scaling of synaptic conductances (defaults to 2)
--sparse: Use sparse rather than dense data structure to represent connectivity

An example invocation of generate_run using these defaults and recording results with a base name of `test':

generate_run.exe test

for Windows users, or:

./generate_run test

for Linux, Mac and other UNIX users. 

This will generate a network of 100 Poisson neurons with 20 Hz firing rate
connected to 10 Izhikevich neurons with a 0.5 probability. 
The same network with sparse connectivity can be used by adding the --sparse flag to the command line.

Another example of an invocation that runs the simulation using the CPU rather than GPU, 
records timing information and uses sparse connectivity would be: 

generate_run.exe --cpu-only --timing --sparse test

for Windows users, or:

./generate_run --cpu-only --timing --sparse test

for Linux, Mac and other UNIX users. 
