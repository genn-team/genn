
  Pulse-coupled Izhikevich network
  ================================

This example model is inspired by simple thalamo-cortical network of Izhikevich 
with an excitatory and an inhibitory population of spiking neurons that are
randomly connected. It creates a pulse-coupled network with 80% excitatory 20%
inhibitory connections, each connecting to nConn neurons with sparse connectivity.

To compile it, navigate to genn/userproject/Izh_sparse_project and type:
  nmake /f WINmakefile
for Windows users, or:
  make
for Linux, Mac and other UNIX users.


  USAGE
  -----

  ./generate_run <0(CPU)/1(GPU)/n(GPU n-2)> <nNeurons> <nConn> <gScale> <outdir> <model name> <input factor>

Mandatory arguments:
CPU/GPU: Choose whether to run the simulation on CPU (`0`), auto GPU (`1`), or GPU (n-2) (`n`).
nNeurons: Number of neurons
nConn: Number of connections per neuron
gScale: General scaling of synaptic conductances
outname: The base name of the output location and output files
model name: The name of the model to execute, as provided this would be `Izh_sparse`

Optional arguments:
DEBUG=0 or DEBUG=1 (default 0): Whether to run in a debugger
FTYPE=DOUBLE of FTYPE=FLOAT (default FLOAT): What floating point type to use
REUSE=0 or REUSE=1 (default 0): Whether to reuse generated connectivity from an earlier run
CPU_ONLY=0 or CPU_ONLY=1 (default 0): Whether to compile in (CUDA independent) "CPU only" mode.

An example invocation of generate_run is:

  ./generate_run 1 10000 1000 1 outdir Izh_sparse 1.0

This would create a pulse coupled network of 8000 excitatory 2000 inhibitory
Izhikevich neurons, each making 1000 connections with other neurons, generating
a mixed alpha and gamma regime. For larger input factor, there is more
input current and more irregular activity, for smaller factors less
and less and more sparse activity. The synapses are of a simple pulse-coupling
type. The results of the simulation are saved in the directory `outdir_output`,
debugging is switched off, and the connectivity is generated afresh (rather than
being read from existing files).

If connectivity were to be read from files, the connectivity files would have to
be in the `inputfiles` sub-directory and be named according to the names of the
synapse populations involved, e.g., `gIzh_sparse_ee` (\<variable name>=`g`
\<model name>=`Izh_sparse` \<synapse population>=`_ee`). These name conventions
are not part of the core GeNN definitions and it is the privilege (or burden)
of the user to find their own in their own versions of `generate_run`.

Another example of an invocation would be: 
  ./generate_run 0 10000 1000 1 outdir Izh_sparse 1.0 FTYPE=DOUBLE DEBUG=0 CPU_ONLY=1
