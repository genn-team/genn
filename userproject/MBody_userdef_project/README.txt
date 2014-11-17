
  Locust olfactory system (Nowotny et al. 2005) with user-defined synapses
  ========================================================================

This is basically same as the MBody1 example, but it redefines built-in synapse models 
as  user-defined models. Also sparse connectivity is used instead of dense.

This example project contains a helper executable called "generate_run", which also
prepares additional synapse connectivity and input pattern data, before compiling and
executing the model. To compile it, simply type:
  nmake /f WINmakefile
for Windows users, or:
  make
for Linux, Mac and other UNIX users. 


  USAGE
  -----

  ./generate_run [CPU/GPU] [nAL] [nKC] [nLH] [nDN] [gscale] [DIR] [MODEL] [DEBUG OFF/ON] [SKIP GENERATION OFF/ONN]

An example invocation of generate_run is:

  ./generate_run 1 100 1000 20 100 0.0025 outname MBody_userdef 0 0

Such a command would generate a locust olfaction model with 100 antennal lobe neurons,
1000 mushroom body Kenyon cells, 20 lateral horn interneurons and 100 mushroom body
output neurons, and launch a simulation of it on a CUDA-enabled GPU. All output files
will be prefixed with "outname" and will be created under the "outname" directory.

In more details, what generate_run program does is: 
a) use some other tools to generate the appropriate connectivity
   matrices and store them in files.

b) build the source code for the model by writing neuron numbers into
   userproject/include/sizes.h, and executing "buildmodel.sh MBody_userdef [DEBUG OFF/ON]".  

c) compile the generated code by invoking "make clean && make" 
   running the code, e.g. "./classol_sim r1 1".


  MODEL INFORMATION
  -----------------

For information regarding the locust olfaction model implemented in this example project, see:

T. Nowotny, R. Huerta, H. D. I. Abarbanel, and M. I. Rabinovich Self-organization in the
olfactory system: One shot odor recognition in insects, Biol Cyber, 93 (6): 436-446 (2005),
doi:10.1007/s00422-005-0019-7 
