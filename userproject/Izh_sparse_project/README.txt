
  Pulse-coupled Izhikevich network
  ================================

This example creates a pulse-coupled network with 80% excitatory 20% inhibitory
connections, each connecting to nConn neurons with sparse connectivity. To compile
it, simply type:
  nmake /f WINmakefile
for Windows users, or:
  make
for Linux, Mac and other UNIX users.


  USAGE
  -----

  ./generate_run <0(CPU)/1(GPU)> <nNeurons> <nConn> <gscale> <outdir> <model name> <debug OFF/ON> <ftype DOUBLE/FLOAT> <use previous connectivity 0(OFF)/1(ON)>
<input factor>

An example invocation of generate_run is:

  ./generate_run 1 10000 1000 1 outdir Izh_sparse 0 FLOAT 0 1.0

This would create a pulse coupled network of 8000 excitatory 2000 inhibitory
Izhikevich neurons, each making 1000 connections with other neurons, generating
a mixed alpha and gamma regime. For larger input factor, there is more
input current and more irregular activity, for smaller factors less
and less and more sparse activity.
