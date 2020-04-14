
Izhikevich network with delayed synapses
========================================

This example project demonstrates the synaptic delay feature of GeNN. It creates
a network of three Izhikevich neuron groups, connected all-to-all with fast, medium
and slow synapse groups. Neurons in the output group only spike if they are
simultaneously innervated by the input neurons, via slow synapses, and the
interneurons, via faster synapses. 


COMPILE (WINDOWS)
-----------------

To run this example project, first build the model into CUDA code by typing:

genn-buildmodel.bat SynDelay.cc

then compile the project by typing:

msbuild SynDelay.sln /t:SynDelay /p:Configuration=Release


COMPILE (MAC AND LINUX)
-----------------------

To run this example project, first build the model into CUDA code by typing:

genn-buildmodel.sh SynDelay.cc

then compile the project by typing:

make


USAGE
-----

syn_delay [directory to save output]
