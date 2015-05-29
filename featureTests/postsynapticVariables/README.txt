
  Feature tests for acessing neuron variables from synapses 
  ========================================

This set of feature tests checks whether the access to neuron
variables from synapses works as expected. 
Tests:
postVarsInSimCode: 
Tests whether post-synaptic neuron variables are
accessible fromt eh "simCode" snippet in synapses;

postVarsInSynapseDynamics: 
Tests whether post-synaptic neuron variables are
accessible fromt the "synapseDynamics" snippet in synapses;

postVarsInPostLearn:
Tests whether post-synaptic variables are accessed correctly in the
simLearnPost code snippet in synapses.

postVarsInSimCode_sparse: 
as postVarsInSimCode but for sparse connectivity.

postVarsInSynapseDynamics_sparse: 
As postVarsInSynapseDynamics but for sparse connectivity.

postVarsInPostLearn_sparse:
As postVarsInPostLearn but for sparse connectivity.


  COMPILE (WINDOWS)
  -----------------

To run this example project, first build the model into CUDA code by typing:

  buildmodel.bat <NN>

where <NN> is the name of the test, e.g. postVarsInSimCode

then compile the project by typing:

  nmake /f WINmakefile<NN> clean
  nmake /f WINmakefile<NN>


  COMPILE (MAC AND LINUX)
  -----------------------

To run a single test, first build the model into CUDA code by typing:

  buildmodel.sh <NN>

where <NN> is the name of the test, e.g. postVarsInSimCode

then compile the test by typing:

  make -f Makefile<NN> clean 
  make -f Makefile<NN> 


  USAGE
  -----

  ./test<NN> [CPU = 0 / GPU = 1] [directory to save output] [write verbose
  debug files 1/0]

If you just want to run all tests, type
   bash runTests.sh

To clean the testing directory, use 
   bash cleanTests.sh


