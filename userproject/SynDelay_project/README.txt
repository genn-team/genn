
  Izhikevich network with delayed synapses
  ========================================

This example project demonstrates the delayed synapse feature of GeNN. It creates
a network of three Izhikevich neuron groups, connected all-to-all with fast, medium
and slow synapse groups. Neurons in the output group only spike if they are
simultaneously innervated by the input neurons, via slow synapses, and the
interneurons, via faster synapses. 


  COMPILE (WINDOWS)
  -----------------

To run this example project, first build the model into CUDA code by typing:

  buildmodel.bat SynDelay

then compile the project by typing:

  nmake /f WINmakefile clean
  nmake /f WINmakefile


  COMPILE (MAC AND LINUX)
  -----------------------

To run this example project, first build the model into CUDA code by typing:

  buildmodel.sh SynDelay

then compile the project by typing:

  make clean && make release


  USAGE
  -----

  ./bin/syn_delay [CPU = 0 / GPU = 1] [directory to save output]
