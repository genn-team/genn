
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

  ./generate_run <0(CPU)/1(GPU)/2..n(GPU0..n-2)> <nNeurons> <nConn> <gscale> <outdir> <model name> <input factor>

An example invocation of generate_run is:

  ./generate_run 1 10000 1000 1 outdir Izh_sparse 1.0

This would create a pulse coupled network of 8000 excitatory 2000 inhibitory
Izhikevich neurons, each making 1000 connections with other neurons, generating
a mixed alpha and gamma regime. For larger input factor, there is more
input current and more irregular activity, for smaller factors less
and less and more sparse activity.

Another example of an invocation would be: 
  ./generate_run 0 10000 1000 1 outdir Izh_sparse 1.0 FTYPE=DOUBLE DEBUG=0 CPU_ONLY=1
