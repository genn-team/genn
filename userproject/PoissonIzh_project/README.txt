
Izhikevich network receiving Poisson input spike trains
=======================================================

In this example project there is again a pool of non-connected Izhikevich model neurons
that are connected to a pool of Poisson input neurons with a fixed probability.
This example project contains a helper executable called "generate_run", which also
prepares additional synapse connectivity and input pattern data, before compiling and
executing the model.

To compile it, navigate to genn/userproject/PoissonIzh_project and type:

nmake /f WINmakefile

for Windows users, or:

make

for Linux, Mac and other UNIX users. 


USAGE
-----

generate_run <0(CPU)/1(GPU)> <nPoisson> <nIzhikevich> <pConn> <gscale> <DIR> <MODEL>

Optional arguments:
DEBUG=0 or DEBUG=1 (default 0): Whether to run in a debugger
FTYPE=DOUBLE or FTYPE=FLOAT (default FLOAT): What floating point type to use
REUSE=0 or REUSE=1 (default 0): Whether to reuse generated connectivity from an earlier run
CPU_ONLY=0 or CPU_ONLY=1 (default 0): Whether to compile in (CUDA independent) "CPU only" mode.

An example invocation of generate_run is:

generate_run.exe 1 100 10 0.5 2 outdir PoissonIzh

for Windows users, or:

./generate_run 1 100 10 0.5 2 outdir PoissonIzh

for Linux, Mac and other UNIX users. 

This will generate a network of 100 Poisson neurons with 20 Hz firing rate
connected to 10 Izhikevich neurons with a 0.5 probability. 
The same network with sparse connectivity can be used by adding
the synapse population with sparse connectivity in PoissonIzh.cc and by uncommenting
the lines following the "//SPARSE CONNECTIVITY" tag in PoissonIzh.cu and commenting the
lines following `//DENSE CONNECTIVITY`.

Another example of an invocation would be: 

generate_run.exe 0 100 10 0.5 2 outdir PoissonIzh FTYPE=DOUBLE CPU_ONLY=1

for Windows users, or:

./generate_run 0 100 10 0.5 2 outdir PoissonIzh FTYPE=DOUBLE CPU_ONLY=1

for Linux, Mac and other UNIX users. 
