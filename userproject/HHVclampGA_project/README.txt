
Genetic algorithm for tracking parameters in a HH model cell
============================================================

This example simulates a population of Hodgkin-Huxley neuron models using GeNN and evolves them with a simple 
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

generate_run [OPTIONS] <outname> 

Mandatory arguments:
outname: The base name of the output location and output files

Optional arguments:
--debug: Builds a debug version of the simulation and attaches the debugger
--cpu-only: Uses CPU rather than CUDA backend for GeNN
--timing: Uses GeNN's timing mechanism to measure performance and displays it at the end of the simulation
--ftype: Sets the floating point precision of the model to either float or double (defaults to float)
--gpu-device: Sets which GPU device to use for the simulation (defaults to -1 which picks automatically)
--protocol: Which changes to apply during the run to the parameters of the "true cell" (defaults to -1 which makes no changes)
--num-pops: Number of neurons in the tracking population (defaults to 12)
--total-time: Time in ms how long to run the simulation  (defaults to 200000ms)

An example invocation of generate_run is:

generate_run.exe test1

for Windows users, or:

./generate_run test1

for Linux, Mac and other UNIX users.

This will simulate 12 Hodgkin-Huxley neurons on the GPU which will for 200000ms ms be matched to a
Hodgkin-Huxley neuron. The output files will be written into a directory of the name test1_output, 
which will be created if it does not yet exist.

Another example of an invocation would be: 

generate_run.exe --cpu-only --timing --protocol 

for Windows users, or:

./generate_run 0 -1 12 200000 test1 FTYPE=DOUBLE CPU_ONLY=1

for Linux, Mac and other UNIX users.
