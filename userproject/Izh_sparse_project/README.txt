
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

  ./generate_run [CPU/GPU] [nNeurons] [nConn] [gscale] [outdir] [model name] [debug OFF/ON] [use previous connectivity OFF/ON]

An example invocation of generate_run is:

  ./generate_run 1 10000 1000 1 outdir Izh_sparse 0 0
