.. index:: pair: page; Examples
.. _doxid-d9/d61/Examples:

Examples
========

GeNN comes with a number of complete examples. At the moment, there are seven such example projects provided with GeNN.



.. _doxid-d9/d61/Examples_1Ex_OneComp:

Single compartment Izhikevich neuron(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

	
	Izhikevich neuron(s) without any connections
	=====================================
	
	This is a minimal example, with only one neuron population (with more or less
	neurons depending on the command line, but without any synapses). The neurons
	are Izhikevich neurons with homogeneous parameters across the neuron population.
	This example project contains a helper executable called "generate_run", 
	which compiles and executes the model.
	
	To compile it, navigate to genn/userproject/OneComp_project and type:
	
	msbuild ..\userprojects.sln /t:generate_one_comp_runner /p:Configuration=Release
	
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
	--num-neurons: Number of neurons to simulate (defaults to 1)
	
	For a first minimal test, using these defaults and recording results with a base name of `test',the system may be used with:
	
	generate_run.exe test
	
	for Windows users, or:
	
	./generate_run test
	
	for Linux, Mac and other UNIX users. 
	
	This would create a set of tonic spiking Izhikevich neurons with no connectivity, 
	receiving a constant identical 4 nA input.
	
	Another example of an invocation that runs the simulation using the CPU rather than GPU, 
	records timing information and 4 neurons would be: 
	
	generate_run.exe --cpu-only --timing --num_neurons=4 test
	
	for Windows users, or:
	
	./generate_run --cpu-only --timing --num_neurons=4 test
	
	for Linux, Mac and other UNIX users.

Izhikevich neuron model: :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`





.. _doxid-d9/d61/Examples_1ex_poissonizh:

Izhikevich neurons driven by Poisson input spike trains:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

	
	Izhikevich network receiving Poisson input spike trains
	=======================================================
	
	In this example project there is again a pool of non-connected Izhikevich model neurons
	that are connected to a pool of Poisson input neurons with a fixed probability.
	This example project contains a helper executable called "generate_run", which compiles and
	executes the model.
	
	To compile it, navigate to genn/userproject/PoissonIzh_project and type:
	
	msbuild ..\userprojects.sln /t:generate_poisson_izh_runner /p:Configuration=Release
	
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
	--num-poisson:  Number of Poisson sources to simulate (defaults to 100)
	--num-izh: Number of Izhikievich neurons to simulate (defaults to 10)
	--pconn: Probability of connection between each pair of poisson sources and neurons (defaults to 0.5)
	--gscale: Scaling of synaptic conductances (defaults to 2)
	--sparse: Use sparse rather than dense data structure to represent connectivity
	
	An example invocation of generate_run using these defaults and recording results with a base name of `test':
	
	generate_run.exe test
	
	for Windows users, or:
	
	./generate_run test
	
	for Linux, Mac and other UNIX users. 
	
	This will generate a network of 100 Poisson neurons with 20 Hz firing rate
	connected to 10 Izhikevich neurons with a 0.5 probability. 
	The same network with sparse connectivity can be used by adding the --sparse flag to the command line.
	
	Another example of an invocation that runs the simulation using the CPU rather than GPU, 
	records timing information and uses sparse connectivity would be: 
	
	generate_run.exe --cpu-only --timing --sparse test
	
	for Windows users, or:
	
	./generate_run --cpu-only --timing --sparse test
	
	for Linux, Mac and other UNIX users.

Izhikevich neuron model: :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`





.. _doxid-d9/d61/Examples_1ex_izhnetwork:

Pulse-coupled Izhikevich network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

	
	Pulse-coupled Izhikevich network
	================================
	
	This example model is inspired by simple thalamo-cortical network of Izhikevich 
	with an excitatory and an inhibitory population of spiking neurons that are
	randomly connected. It creates a pulse-coupled network with 80% excitatory 20%
	inhibitory connections, each connecting to a fixed number of neurons with sparse connectivity.
	
	To compile it, navigate to genn/userproject/Izh_sparse_project and type:
	
	msbuild ..\userprojects.sln /t:generate_izh_sparse_runner /p:Configuration=Release
	
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
	--num-neurons: Number of neurons (defaults to 10000)
	--num-connections: Number of connections per neuron (defaults to 1000)
	--gscale: General scaling of synaptic conductances (defaults to 1.0)
	
	An example invocation of generate_run using these defaults and recording results with a base name of `test' would be:
	
	generate_run.exe test
	
	for Windows users, or:
	
	./generate_run test
	
	for Linux, Mac and other UNIX users.
	
	This would create a pulse coupled network of 8000 excitatory 2000 inhibitory
	Izhikevich neurons, each making 1000 connections with other neurons, generating
	a mixed alpha and gamma regime. For larger input factor, there is more
	input current and more irregular activity, for smaller factors less
	and less and more sparse activity. The synapses are of a simple pulse-coupling
	type. The results of the simulation are saved in the directory `outdir_output`.
	
	Another example of an invocation that runs the simulation using the CPU rather than GPU, 
	records timing information and doubles the number of neurons would be: 
	
	generate_run.exe --cpu-only --timing --num_neurons=20000 test
	
	for Windows users, or:
	
	./generate_run --cpu-only --timing --num_neurons=20000 test
	
	for Linux, Mac and other UNIX users.

Izhikevich neuron model: :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`





.. _doxid-d9/d61/Examples_1ex_izhdelay:

Izhikevich network with delayed synapses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

	
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

Izhikevich neuron model: :ref:`izhikevich2003simple <doxid-d0/de3/citelist_1CITEREF_izhikevich2003simple>`





.. _doxid-d9/d61/Examples_1ex_mbody:

Insect olfaction model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

	
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

Nowotny insect olfaction model: :ref:`nowotny2005self <doxid-d0/de3/citelist_1CITEREF_nowotny2005self>`; Traub-Miles Hodgkin-Huxley neuron model: :ref:`Traub1991 <doxid-d0/de3/citelist_1CITEREF_Traub1991>`





.. _doxid-d9/d61/Examples_1ex_Vclamp:

Voltage clamp simulation to estimate Hodgkin-Huxley parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

	
	Genetic algorithm for tracking parameters in a HH model cell
	============================================================
	
	This example simulates a population of Hodgkin-Huxley neuron models using GeNN and evolves them with a simple 
	guided random search (simple GA) to mimic the dynamics of a separate Hodgkin-Huxley
	neuron that is simulated on the CPU. The parameters of the CPU simulated "true cell" are drifting 
	according to a user-chosen protocol: Either one of the parameters gNa, ENa, gKd, EKd, gleak,
	Eleak, Cmem are modified by a sinusoidal addition (voltage parameters) or factor (conductance or capacitance) - 
	protocol 0-6. For protocol 7 all 7 parameters undergo a random walk concurrently.
	
	To compile it, navigate to genn/userproject/HHVclampGA_project and type:
	
	msbuild ..\userproject.sln /t:generate_hhvclamp_runner /p:Configuration=Release
	
	for Windows users, or:s
	
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
	--protocol: Which changes to apply during the run to the parameters of the "true cell" (defaults to -1 which makes no changes)
	--num-pops: Number of neurons in the tracking population (defaults to 5000)
	--total-time: Time in ms how long to run the simulation  (defaults to 1000 ms)
	
	An example invocation of generate_run is:
	
	generate_run.exe test1
	
	for Windows users, or:
	
	./generate_run test1
	
	for Linux, Mac and other UNIX users.
	
	This will simulate 5000 Hodgkin-Huxley neurons on the GPU which will, for 1000 ms, be matched to a
	Hodgkin-Huxley neuron. The output files will be written into a directory of the name test1_output, 
	which will be created if it does not yet exist.
	
	Another example of an invocation that records timing information for the the simulation and runs it for 10000 ms would be: 
	
	generate_run.exe --timing --total-time 10000
	
	for Windows users, or:
	
	./generate_run --timing --total-time 10000
	
	for Linux, Mac and other UNIX users.

Traub-Miles Hodgkin-Huxley neuron model: :ref:`Traub1991 <doxid-d0/de3/citelist_1CITEREF_Traub1991>`





.. _doxid-d9/d61/Examples_1ex_Schmuker:

A neuromorphic network for generic multivariate data classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

	Author: Alan Diamond, University of Sussex, 2014
	
	This project recreates using GeNN the spiking classifier design used in the paper
	
	"A neuromorphic network for generic multivariate data classification"
	 Authors: Michael Schmuker, Thomas Pfeil, Martin Paul Nawrota
	    
	The classifier design is based on an abstraction of the insect olfactory system.
	This example uses the IRIS stadard data set as a test for the classifier
	
	BUILD / RUN INSTRUCTIONS 
	
	Install GeNN from the internet released build, following instruction on setting your PATH etc
	
	Start a terminal session
	
	cd to this project directory (userproject/Model_Schmuker_2014_project)
	
	To build the model using the GENN meta compiler type:
	
	genn-buildmodel.sh Model_Schmuker_2014_classifier.cc
	
	for Linux, Mac and other UNIX systems, or:
	
	genn-buildmodel.bat Model_Schmuker_2014_classifier.cc
	
	for Windows systems (add -d for a debug build).
	
	You should only have to do this at the start, or when you change your actual network model  (i.e. editing the file Model_Schmuker_2014_classifier.cc )
	
	Then to compile the experiment plus the GeNN created C/CUDA code type:-
	
	make
	
	for Linux, Mac and other UNIX users (add DEBUG=1 if using debug mode), or:
	
	msbuild Schmuker2014_classifier.vcxproj /p:Configuration=Release
	
	for Windows users (change Release to Debug if using debug mode).
	
	Once it compiles you should be able to run the classifier against the included Iris dataset.
	
	type
	
	./experiment .
	
	for Linux, Mac and other UNIX systems, or:
	
	Schmuker2014_classifier .
	
	for Windows systems.
	
	This is how it works roughly.
	The experiment (experiment.cu) controls the experiment at a high level. It mostly does this by instructing the classifier (Schmuker2014_classifier.cu) which does the grunt work.
	
	So the experiment first tells the classifier to set up the GPU with the model and synapse data.
	
	Then it chooses the training and test set data.
	
	It runs through the training set , with plasticity ON , telling the classifier to run with the specfied observation and collecting the classifier decision.
	
	Then it runs through the test set with plasticity OFF  and collects the results in various reporting files.
	
	At the highest level it also has a loop where you can cycle through a list of parameter values e.g. some threshold value for the classifier to use. It will then report on the performance for each value. You should be aware that some parameter changes won't actually affect the classifier unless you invoke a re-initialisation of some sort. E.g. anything to do with VRs will require the input data cache to be reset between values, anything to do with non-plastic synapse weights won't get cleared down until you upload a changed set to the GPU etc.
	
	You should also note there is no option currently to run on CPU, this is not due to the demanding task, it just hasn't been tweaked yet to allow for this (small change).



:ref:`Previous <doxid-d7/d98/Quickstart>` \| :ref:`Top <doxid-d9/d61/Examples>` \| :ref:`Next <doxid-d2/dba/SpineML>`

