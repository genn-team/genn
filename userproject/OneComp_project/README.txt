
  Izhikevich neuron(s) without any connections
  =====================================

This example project contains a helper executable called "generate_run", which also
prepares additional synapse connectivity and input pattern data, before compiling and
executing the model. To compile it, simply type:
  nmake /f WINmakefile
for Windows users, or:
  make
for Linux, Mac and other UNIX users. 


  USAGE
  -----

  ./generate_run <0(CPU)/1(GPU)> <n> <DIR> <MODEL>
and optional arguments:
DEBUG=0 or DEBUG=1 (default 0): Whether to run in a debugger
FTYPE=DOUBLE of FTYPE=FLOAT (default FLOAT): What floating point type to use
REUSE=0 or REUSE=1 (default 0): Whether to reuse generated connectivity from an earlier run
CPU_ONLY=0 or CPU_ONLY=1 (default 0): Whether to compile in (CUDA independent) "CPU only" mode.

For a first minimal test, the system may be used with:

  ./generate_run 1 1 outdir OneComp 

This would create a set of tonic spiking Izhikevich neurons with no connectivity, 
receiving a constant identical 4 nA input. It is lso possible to use the model
with a sinusoidal input instead, by setting the input to INPRULE.

Another example of an invocation would be: 
  ./generate_run 0 1 outdir OneComp FTYPE=DOUBLE CPU_ONLY=1
