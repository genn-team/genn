
Locust olfactory system (Nowotny et al. 2005)
=============================================

This project implements the insect olfaction model by Nowotny et al. that demonstrates 
self-organized clustering of odours in a simulation of the insect antennal lobe and 
mushroom body. As provided, the model works with conductance based Hodgkin-Huxley neurons 
and several different synapse types, conductance based (but pulse-coupled) excitatory 
synapses, graded inhibitory synapses and synapses with a simplified STDP rule.  
This example project contains a helper executable called "generate_run", which 
prepares input pattern data, before compiling and executing the model.

To compile it, navigate to genn/userproject/MBody1_project and type:

msbuild ..\userprojects.sln /t:generate_mbody1_runner /p:Configuration=Release

for Windows users, or:

make

for Linux, Mac and other UNIX users. 


USAGE
-----

generate_run [OPTIONS] <outname> 

Mandatory arguments:
outname: The base name of the output location and output files

Optional arguments:
--debug: Builds a debug version of the simulation and attaches the debugger
--cpu-only: Uses CPU rather than CUDA backend for GeNN
--timing: Uses GeNN's timing mechanism to measure performance and displays it at the end of the simulation
--ftype: Sets the floating point precision of the model to either float or double (defaults to float)
--gpu-device: Sets which GPU device to use for the simulation (defaults to -1 which picks automatically)
--num-al: Number of neurons in the antennal lobe (AL), the input neurons to this model (defaults to 100)
--num-kc: Number of Kenyon cells (KC) in the "hidden layer" (defaults to 1000)
--num-lhi: Number of lateral horn interneurons, implementing gain control (defaults to 20)
--num-dn: Number of decision neurons (DN) in the output layer (defaults to 100)
--gscale: A general rescaling factor for synaptic strength (defaults to 0.0025)
--bitmask: Use bitmasks to represent sparse PN->KC connectivity rather than dense connectivity
--delayed-synapses: Rather than use constant delays of DT throughough, use delays of (5 * DT) ms on KC->DN and 
of (3 * DT) ms on DN->DN synapse populations

An example invocation of generate_run using these defaults and recording results with a base name of `test' would be:

generate_run.exe test

for Windows users, or:

./generate_run test

for Linux, Mac and other UNIX users. 

Such a command would generate a locust olfaction model with 100 antennal lobe neurons,
1000 mushroom body Kenyon cells, 20 lateral horn interneurons and 100 mushroom body
output neurons, and launch a simulation of it on a CUDA-enabled GPU using single
precision floating point numbers. All output files will be prefixed with "test"
and will be created under the "test" directory. The model that is run is defined
in `model/MBody1.cc`, debugging is switched off and the model would be simulated using
float (single precision floating point) variables.

In more details, what generate_run program does is: 
a) use other tools to generate input patterns.

b) build the source code for the model by writing neuron numbers into
   ./model/sizes.h, and executing "genn-buildmodel.sh ./model/MBody1.cc.

c) compile the generated code by invoking "make clean && make" 
   running the code, e.g. "./classol_sim r1".

Another example of an invocation that runs the simulation using CPU rather than GPU, 
records timing information and uses bitmask connectivity would be: 

generate_run.exe --cpu-only --timing --bitmask test

for Windows users, or:

./generate_run --cpu-only --timing --bitmask test

for Linux, Mac and other UNIX users.

As provided, the model outputs  `test.dn.st', `test.kc.st', `test.lhi.st' and `test.pn.st' files 
which contain the spiking activity observed in each population in the simulation. There are two
columns in this ASCII file, the first one containing the time of a spike and the second one, 
the ID of the neuron that spiked. MATLAB users can use the scripts in the `matlab` directory to plot
the results of a simulation and Python users can use the plot_spikes.py script in userproject/python. 
For more about the model itself and the scientific insights gained from it, see Nowotny et al. referenced below.


MODEL INFORMATION
-----------------

For information regarding the locust olfaction model implemented in this example project, see:

T. Nowotny, R. Huerta, H. D. I. Abarbanel, and M. I. Rabinovich Self-organization in the
olfactory system: One shot odor recognition in insects, Biol Cyber, 93 (6): 436-446 (2005),
doi:10.1007/s00422-005-0019-7 
=======

Locust olfactory system (Nowotny et al. 2005)
=============================================

This project implements the insect olfaction model by Nowotny et
al. that demonstrates self-organized clustering of odours in a
simulation of the insect antennal lobe and mushroom body. As provided
the model works with conductance based Hodgkin-Huxley neurons and
several different synapse types, conductance based (but pulse-coupled)
excitatory synapses, graded inhibitory synapses and synapses with a
simplified STDP rule. This example project contains a helper executable called "generate_run", which 
prepares input pattern data, before compiling and
executing the model.

To compile it, navigate to genn/userproject/MBody1_project and type:

msbuild ..\userprojects.sln /t:generate_mbody1_runner /p:Configuration=Release

for Windows users, or:

make

for Linux, Mac and other UNIX users. 


USAGE
-----

generate_run [OPTIONS] <outname> 

Mandatory arguments:
outname: The base name of the output location and output files

Optional arguments:
--debug: Builds a debug version of the simulation and attaches the debugger
--cpu-only: Uses CPU rather than CUDA backend for GeNN
--timing: Uses GeNN's timing mechanism to measure performance and displays it at the end of the simulation
--ftype: Sets the floating point precision of the model to either float or double (defaults to float)
--gpu-device: Sets which GPU device to use for the simulation (defaults to -1 which picks automatically)
--num-al: Number of neurons in the antennal lobe (AL), the input neurons to this model (defaults to 100)
--num-kc: Number of Kenyon cells (KC) in the "hidden layer" (defaults to 1000)
--num-lhi: Number of lateral horn interneurons, implementing gain control (defaults to 20)
--num-dn: Number of decision neurons (DN) in the output layer (defaults to 100)
--gscale: A general rescaling factor for synaptic strength (defaults to 0.0025)
--bitmask: Use bitmasks to represent sparse PN->KC connectivity rather than dense connectivity
--delayed-synapses: Rather than use constant delays of DT throughough, use delays of (5 * DT) ms on KC->DN and of (3 * DT) ms on DN->DN synapse populations

An example invocation of generate_run using these defaults and recording results with a base name of `test' would be:

generate_run.exe test

for Windows users, or:

./generate_run test

for Linux, Mac and other UNIX users. 

Such a command would generate a locust olfaction model with 100 antennal lobe neurons,
1000 mushroom body Kenyon cells, 20 lateral horn interneurons and 100 mushroom body
output neurons, and launch a simulation of it on a CUDA-enabled GPU using single
precision floating point numbers. All output files will be prefixed with "test"
and will be created under the "test" directory. The model that is run is defined
in `model/MBody1.cc`, debugging is switched off and the model would be simulated using
float (single precision floating point) variables.

In more details, what generate_run program does is: 
a) use another tools to generate input patterns.

b) build the source code for the model by writing neuron numbers into
   ./model/sizes.h, and executing "genn-buildmodel.sh ./model/MBody1.cc.

c) compile the generated code by invoking "make clean && make" 
   running the code, e.g. "./classol_sim r1".

Another example of an invocation that runs the simulation using the CPU rather than GPU, 
records timing information and uses bitmask connectivity would be: 

generate_run.exe --cpu-only --timing --bitmask test

for Windows users, or:

./generate_run --cpu-only --timing --bitmask test

for Linux, Mac and other UNIX users.

As provided, the model outputs  `test.dn.st', `test.kc.st', `test.lhi.st' and `test.pn.st' files which contain
the spiking activity observed in each population inthe simulation, There are two
columns in this ASCII file, the first one containing the time of
a spike and the second one the ID of the neuron that spiked. Users
of matlab can use the scripts in the `matlab` directory to plot
the results of a simulation and users of python can use the plot_spikes.py script in userproject/python. 
For more about the model itself and the scientific insights gained from it see Nowotny et al. referenced below.


MODEL INFORMATION
-----------------

For information regarding the locust olfaction model implemented in this example project, see:

T. Nowotny, R. Huerta, H. D. I. Abarbanel, and M. I. Rabinovich Self-organization in the
olfactory system: One shot odor recognition in insects, Biol Cyber, 93 (6): 436-446 (2005),
doi:10.1007/s00422-005-0019-7 