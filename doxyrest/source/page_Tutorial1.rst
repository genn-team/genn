.. index:: pair: page; Tutorial 1
.. _doxid-d5/dbb/Tutorial1:

Tutorial 1
==========

In this tutorial we will go through step by step instructions how to create and run your first GeNN simulation from scratch.



.. _doxid-d5/dbb/Tutorial1_1ModelDefinition:

The Model Definition
~~~~~~~~~~~~~~~~~~~~

In this tutorial we will use a pre-defined Hodgkin-Huxley neuron model (:ref:`NeuronModels::TraubMiles <doxid-d2/dc3/classNeuronModels_1_1TraubMiles>`) and create a simulation consisting of ten such neurons without any synaptic connections. We will run this simulation on a GPU and save the results - firstly to stdout and then to file.

The first step is to write a model definition function in a model definition file. Create a new directory and, within that, create a new empty file called ``tenHHModel.cc`` using your favourite text editor, e.g.

.. ref-code-block:: cpp

	>> emacs tenHHModel.cc &

The ">>" in the example code snippets refers to a shell prompt in a unix shell, do not enter them as part of your shell commands.

The model definition file contains the definition of the network model we want to simulate. First, we need to include the GeNN model specification code ``modelSpec.h``. Then the model definition takes the form of a function named ``modelDefinition`` that takes one argument, passed by reference, of type ``:ref:`ModelSpec <doxid-da/dfd/classModelSpec>```. Type in your ``tenHHModel.cc`` file:

.. ref-code-block:: cpp

	// Model definintion file tenHHModel.cc
	
	#include "modelSpec.h"
	
	void modelDefinition(:ref:`ModelSpec <doxid-da/dfd/classModelSpec>` &model)
	{
	    // definition of tenHHModel
	}

Two standard elements to the `modelDefinition function are setting the simulation step size and setting the name of the model:

.. ref-code-block:: cpp

	model.:ref:`setDT <doxid-da/dfd/classModelSpec_1a329236a3b07044b82bfda5b4f741d8e1>`(0.1);
	model.:ref:`setName <doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4>`("tenHHModel");

With this we have fixed the integration time step to ``0.1`` in the usual time units. The typical units in GeNN are ``ms``, ``mV``, ``nF``, and ``S``. Therefore, this defines ``DT= 0.1 ms``.

Making the actual model definition makes use of the :ref:`ModelSpec::addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>` and :ref:`ModelSpec::addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>` member functions of the :ref:`ModelSpec <doxid-da/dfd/classModelSpec>` object. The arguments to a call to :ref:`ModelSpec::addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>` are

* ``NeuronModel`` : template parameter specifying the neuron model class to use

* ``const std::string &name`` : the name of the population

* ``unsigned int size`` : The number of neurons in the population

* ``const NeuronModel::ParamValues &paramValues`` : Parameter values for the neurons in the population

* ``const NeuronModel::VarValues &varInitialisers`` : Initial values or initialisation snippets for variables of this neuron type

We first create the parameter and initial variable arrays,

.. ref-code-block:: cpp

	// definition of tenHHModel
	:ref:`NeuronModels::TraubMiles::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` p(
	    7.15,       // 0 - gNa: Na conductance in muS
	    50.0,       // 1 - ENa: Na equi potential in mV
	    1.43,       // 2 - gK: K conductance in muS
	    -95.0,      // 3 - EK: K equi potential in mV 
	    0.02672,    // 4 - gl: leak conductance in muS
	    -63.563,    // 5 - El: leak equi potential in mV
	    0.143);     // 6 - Cmem: membr. capacity density in nF
	
	:ref:`NeuronModels::TraubMiles::VarValues <doxid-d6/d24/classModels_1_1VarInitContainerBase>` ini(
	    -60.0,      // 0 - membrane potential V
	    0.0529324,  // 1 - prob. for Na channel activation m
	    0.3176767,  // 2 - prob. for not Na channel blocking h
	    0.5961207); // 3 - prob. for K channel activation n

The comments are obviously only for clarity, they can in principle be omitted. To avoid any confusion about the meaning of parameters and variables, however, we recommend strongly to always include comments of this type.

Having defined the parameter values and initial values we can now create the neuron population,

.. ref-code-block:: cpp

	model.:ref:`addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>`<:ref:`NeuronModels::TraubMiles <doxid-d2/dc3/classNeuronModels_1_1TraubMiles>`>("Pop1", 10, p, ini);

This completes the model definition in this example. The complete ``tenHHModel.cc`` file now should look like this:

.. ref-code-block:: cpp

	// Model definintion file tenHHModel.cc
	
	#include "modelSpec.h"
	
	void modelDefinition(:ref:`ModelSpec <doxid-da/dfd/classModelSpec>` &model)
	{
	    // definition of tenHHModel
	    model.:ref:`setDT <doxid-da/dfd/classModelSpec_1a329236a3b07044b82bfda5b4f741d8e1>`(0.1);
	    model.:ref:`setName <doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4>`("tenHHModel");
	
	    :ref:`NeuronModels::TraubMiles::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` p(
	        7.15,       // 0 - gNa: Na conductance in muS
	        50.0,       // 1 - ENa: Na equi potential in mV
	        1.43,       // 2 - gK: K conductance in muS
	        -95.0,      // 3 - EK: K equi potential in mV 
	        0.02672,    // 4 - gl: leak conductance in muS
	        -63.563,    // 5 - El: leak equi potential in mV
	        0.143);     // 6 - Cmem: membr. capacity density in nF
	
	    :ref:`NeuronModels::TraubMiles::VarValues <doxid-d6/d24/classModels_1_1VarInitContainerBase>` ini(
	        -60.0,      // 0 - membrane potential V
	        0.0529324,  // 1 - prob. for Na channel activation m
	        0.3176767,  // 2 - prob. for not Na channel blocking h
	        0.5961207); // 3 - prob. for K channel activation n
	
	    model.:ref:`addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>`<:ref:`NeuronModels::TraubMiles <doxid-d2/dc3/classNeuronModels_1_1TraubMiles>`>("Pop1", 10, p, ini);
	}

This model definition suffices to generate code for simulating the ten Hodgkin-Huxley neurons on the a GPU or CPU. The second part of a GeNN simulation is the user code that sets up the simulation, does the data handling for input and output and generally defines the numerical experiment to be run.





.. _doxid-d5/dbb/Tutorial1_1buildModel:

Building the model
~~~~~~~~~~~~~~~~~~

To use GeNN to build your model description into simulation code, use a terminal to navigate to the directory containing your ``tenHHModel.cc`` file and, on Linux or Mac, type:

.. ref-code-block:: cpp

	>> genn-buildmodel.sh tenHHModel.cc

Alternatively, on Windows, type:

.. ref-code-block:: cpp

	>> genn-buildmodel.bat tenHHModel.cc

If you don't have an NVIDIA GPU and are running GeNN in CPU_ONLY mode, you can invoke ``genn-buildmodel`` with a ``-c`` option so, on Linux or Mac:

.. ref-code-block:: cpp

	>> genn-buildmodel.sh -c tenHHModel.cc

or on Windows:

.. ref-code-block:: cpp

	>> genn-buildmodel.bat -c tenHHModel.cc

If GeNN has been added to your path and ``CUDA_PATH`` is correctly configured, you should see some compile output ending in ``Model build complete ...``.





.. _doxid-d5/dbb/Tutorial1_1userCode:

User Code
~~~~~~~~~

GeNN will now have generated the code to simulate the model for one timestep using a function ``stepTime()``. To make use of this code, we need to define a minimal C/C++ main function. For the purposes of this tutorial we will initially simply run the model for one simulated second and record the final neuron variables into a file. Open a new empty file ``tenHHSimulation.cc`` in an editor and type

.. ref-code-block:: cpp

	// tenHHModel simulation code
	#include "tenHHModel_CODE/definitions.h"
	
	int main()
	{
	    allocateMem();
	    initialize();
	    return 0;
	}

This boiler plate code includes the header file for the generated code ``definitions.h`` in the subdirectory ``tenHHModel_CODE`` where GeNN deposits all generated code (this corresponds to the name passed to the ``:ref:`ModelSpec::setName <doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4>``` function). Calling ``allocateMem()`` allocates the memory structures for all neuron variables and ``initialize()`` launches a GPU kernel which initialise all state variables to their initial values. Now we can use the generated code to integrate the neuron equations provided by GeNN for 1000ms. To do so, we add after ``initialize();`` The ``t`` variable is provided by GeNN to keep track of the current simulation time in milliseconds.



.. ref-code-block:: cpp

	while (t < 1000.0f) {
	    stepTime();
	}

and we need to copy the result back to the host before outputting it to stdout (this will do nothing if you are running the model on a CPU),

.. ref-code-block:: cpp

	pullPop1StateFromDevice();
	for (int j= 0; j < 10; j++) {  
	    std::cout << VPop1[j] << " ";
	    std::cout << mPop1[j] << " ";
	    std::cout << hPop1[j] << " ";
	    std::cout << nPop1[j] << std::endl;
	}

``pullPop1StateFromDevice()`` copies all relevant state variables of the ``Pop1`` neuron group from the GPU to the CPU main memory. Then we can output the results to stdout by looping through all 10 neurons and outputting the state variables VPop1, mPop1, hPop1, nPop1. The naming convention for variables in GeNN is the variable name defined by the neuron type, here TraubMiles defining V, m, h, and n, followed by the population name, here ``Pop1``.

This completes the user code. The complete ``tenHHSimulation.cc`` file should now look like

.. ref-code-block:: cpp

	// tenHHModel simulation code
	#include "tenHHModel_CODE/definitions.h"
	
	int main()
	{
	    allocateMem();
	    initialize();
	
	    while (t < 1000.0f) {
	        stepTime();
	    }
	    pullPop1StateFromDevice();
	
	    for (int j= 0; j < 10; j++) {  
	        std::cout << VPop1[j] << " ";
	        std::cout << mPop1[j] << " ";
	        std::cout << hPop1[j] << " ";
	        std::cout << nPop1[j] << std::endl;
	    }  
	    return 0;
	}





.. _doxid-d5/dbb/Tutorial1_1BuildingSimUnix:

Building the simulator (Linux or Mac)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Linux and Mac, GeNN simulations are typically built using a simple Makefile which can be generated with the following command:

.. ref-code-block:: cpp

	genn-create-user-project.sh tennHHModel tenHHSimulation.cc

This defines that the model is named tennHHModel and the simulation code is given in the file ``tenHHSimulation.cc`` that we completed above. Now type

.. ref-code-block:: cpp

	make





.. _doxid-d5/dbb/Tutorial1_1BuildingSimWindows:

Building the simulator (Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So that projects can be easily debugged within the Visual Studio IDE (see section :ref:`Debugging suggestions <doxid-d0/da6/UserGuide_1Debugging>` for more details), Windows projects are built using an MSBuild script typically with the same title as the final executable. A suitable solution and project can be generated automatically with the following command:

.. ref-code-block:: cpp

	genn-create-user-project.bat tennHHModel tenHHSimulation.cc

his defines that the model is named tennHHModel and the simulation code is given in the file ``tenHHSimulation.cc`` that we completed above. Now type

.. ref-code-block:: cpp

	msbuild tennHHModel.sln /p:Configuration=Release /t:tennHHModel





.. _doxid-d5/dbb/Tutorial1_1RunningSim:

Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~

You can now execute your newly-built simulator on Linux or Mac with

.. ref-code-block:: cpp

	./tennHHModel

Or on Windows with

.. ref-code-block:: cpp

	tennHHModel_Release

The output you obtain should look like

.. ref-code-block:: cpp

	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243
	-63.7838 0.0350042 0.336314 0.563243





.. _doxid-d5/dbb/Tutorial1_1Input:

Reading
~~~~~~~

This is not particularly interesting as we are just observing the final value of the membrane potentials. To see what is going on in the meantime, we need to copy intermediate values from the device and save them into a file. This can be done in many ways but one sensible way of doing this is to replace the calls to ``stepTime`` in ``tenHHSimulation.cc`` with something like this:

.. ref-code-block:: cpp

	std::ofstream os("tenHH_output.V.dat");
	while (t < 1000.0f) {
	    stepTime();
	
	    pullVPop1FromDevice();
	
	    os << t << " ";
	    for (int j= 0; j < 10; j++) {
	        os << VPop1[j] << " ";
	    }
	    os << std::endl;
	}
	os.close();

t is a global variable updated by the GeNN code to keep track of elapsed simulation time in ms.

we switched from using ``pullPop1StateFromDevice()`` to ``pullVPop1FromDevice()`` as we are now only interested in the membrane voltage of the neuron.

You will also need to add:

.. ref-code-block:: cpp

	#include <fstream>

to the top of tenHHSimulation.cc. After building the model; and building and running the simulator as described above there should be a file ``tenHH_output.V.dat`` in the same directory. If you plot column one (time) against the subsequent 10 columns (voltage of the 10 neurons), you should observe dynamics like this:

.. image:: tenHHexample.png



.. image:: tenHHexample.png
	:alt: width=10cm

However so far, the neurons are not connected and do not receive input. As the :ref:`NeuronModels::TraubMiles <doxid-d2/dc3/classNeuronModels_1_1TraubMiles>` model is silent in such conditions, the membrane voltages of the 10 neurons will simply drift from the -60mV they were initialised at to their resting potential.

:ref:`Previous <doxid-d5/d24/sectSynapseModels>` \| :ref:`Top <doxid-d5/dbb/Tutorial1>` \| :ref:`Next <doxid-dc/d7e/Tutorial2>`

