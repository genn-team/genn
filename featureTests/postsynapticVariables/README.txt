
  Feature tests for acessing neuron variables from synapses 
  ========================================

This set of feature tests checks whether the access to neuron
variables from synapses works as expected. 
Tests:
postVarsInSimCode: 
Tests whether pre-synaptic neuron variables are
accessible fromt eh "simCode" snippet in synapses;

postVarsInSynaoseDynamics: 
Tests whether pre-synaptic neuron variables are
accessible fromt the "synapseDynamics" snippet in synapses;


  COMPILE (WINDOWS)
  -----------------

To run this example project, first build the model into CUDA code by typing:

  buildmodel.bat **

then compile the project by typing:

  nmake /f WINmakefile** clean
  nmake /f WINmakefile**


  COMPILE (MAC AND LINUX)
  -----------------------

To run this example project, first build the model into CUDA code by typing:

  buildmodel.sh **

then compile the project by typing:

  make -f Makefile** clean 
  make -f Makefile** 


  USAGE
  -----

  ./test** [CPU = 0 / GPU = 1] [directory to save output] [write verbose
  debug files 1/0]

(here ** represents the name of the test executable, e.g. PostVarsInSimCode 


