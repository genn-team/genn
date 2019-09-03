.. index:: pair: page; Quickstart
.. _doxid-d7/d98/Quickstart:

Quickstart
==========

GeNN is based on the idea of code generation for the involved GPU or CPU simulation code for neuronal network models but leaves a lot of freedom how to use the generated code in the final application. To facilitate the use of GeNN on the background of this philosophy, it comes with a number of complete examples containing both the model description code that is used by GeNN for code generation and the "user side code" to run the generated model and safe the results. Some of the example models such as the :ref:`Insect olfaction model <doxid-d9/d61/Examples_1ex_mbody>` use an ``generate_run`` executable which automates the building and simulation of the model. Using these executables, running these complete examples should be achievable in a few minutes. The necessary steps are described below.



.. _doxid-d7/d98/Quickstart_1example:

Running an Example Model
~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-d7/d98/Quickstart_1unix_quick:

Unix
----

In order to build the ``generate_run`` executable as well as any additional tools required for the model, open a shell and navigate to the ``userproject/MBody1_project`` directory. Then type

.. ref-code-block:: cpp

	make

to generate an executable that you can invoke with

.. ref-code-block:: cpp

	./generate_run test1

or, if you don't have an NVIDIA GPU and are running GeNN in CPU_ONLY mode, you can instead invoke this executable with

.. ref-code-block:: cpp

	./generate_run --cpu-only test1





.. _doxid-d7/d98/Quickstart_1windows_quick:

Windows
-------

While GeNN can be used from within Visual Studio, in this example we will use a ``cmd`` window. Open a Visual Studio ``cmd`` window via Start: All Programs: Visual Studio: Tools: x86 Native Tools Command Prompt, and navigate to the ``userproject\tools`` directory. Then compile the additional tools and the ``generate_run`` executable for creating and running the project:

.. ref-code-block:: cpp

	msbuild ..\userprojects.sln /t:generate_mbody1_runner /p:Configuration=Release

to generate an executable that you can invoke with

.. ref-code-block:: cpp

	generate_run test1

or, if you don't have an NVIDIA GPU and are running GeNN in CPU_ONLY mode, you can instead invoke this executable with

.. ref-code-block:: cpp

	generate_run --cpu-only test1





.. _doxid-d7/d98/Quickstart_1quick_visualising:

Visualising results
-------------------

These steps will build and simulate a model of the locust olfactory system with default parameters of 100 projection neurons, 1000 Kenyon cells, 20 lateral horn interneurons and 100 output neurons in the mushroom body lobes. If the model isn't build in CPU_ONLY mode it will be simulated on an automatically chosen GPU.

The generate_run tool generates input patterns and writes them to file, compiles and runs the model using these files as inputs and finally output the resulting spiking activity. For more information of the options passed to this command see the :ref:`Insect olfaction model <doxid-d9/d61/Examples_1ex_mbody>` section. The results of the simulation can be plotted with

.. ref-code-block:: cpp

	python plot.py test1

The MBody1 example is already a highly integrated example that showcases many of the features of GeNN and how to program the user-side code for a GeNN application. More details in the :ref:`User Manual <doxid-dc/d05/UserManual>`.







.. _doxid-d7/d98/Quickstart_1how_to:

How to use GeNN for New Projects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating and running projects in GeNN involves a few steps ranging from defining the fundamentals of the model, inputs to the model, details of the model like specific connectivity matrices or initial values, running the model, and analyzing or saving the data.

GeNN code is generally created by passing the C++ model file (see :ref:`below <doxid-d7/d98/Quickstart_1ownmodel>`) directly to the genn-buildmodel script. Another way to use GeNN is to create or modify a script or executable such as ``userproject/MBody1_project/generate_run.cc`` that wraps around the other programs that are used for each of the steps listed above. In more detail, the GeNN workflow consists of:

#. Either use external programs to generate connectivity and input files to be loaded into the user side code at runtime or generate these matrices directly inside the user side code.

#. Generating the model simulation code using ``genn-buildmodel.sh`` (On Linux or Mac) or ``genn-buildmodel.bat`` (on Windows). For example, inside the ``generate_run`` engine used by the MBody1_project, the following command is executed on Linux:
   
   .. ref-code-block:: cpp
   
   	genn-buildmodel.sh MBody1.cc
   
   or, if you don't have an NVIDIA GPU and are running GeNN in CPU_ONLY mode, the following command is executed:
   
   .. ref-code-block:: cpp
   
   	genn-buildmodel.sh -c MBody1.cc
   
   The ``genn-buildmodel`` script compiles the GeNN code generator in conjunction with the user-provided model description ``model/MBody1.cc``. It then executes the GeNN code generator to generate the complete model simulation code for the model.

#. Provide a build script to compile the generated model simulation and the user side code into a simulator executable (in the case of the MBody1 example this consists the file ``MBody1Sim.cc``). On Linux or Mac a suitable GNU makefile can be created by running:
   
   .. ref-code-block:: cpp
   
   	genn-create-user-project.sh MBody1 MBody1Sim.cc
   
   And on Windows an MSBuild project can be created by running:
   
   .. ref-code-block:: cpp
   
   	genn-create-user-project.bat MBody1 MBody1Sim.cc

#. Compile the simulator executable by invoking GNU make on Linux or Mac:
   
   .. ref-code-block:: cpp
   
   	make clean all
   
   or MSbuild on Windows:
   
   .. ref-code-block:: cpp
   
   	msbuild MBody1.sln /t:MBody1 /p:Configuration=Release

#. Finally, run the resulting stand-alone simulator executable. In the MBody1 example, this is called ``MBody1`` on Linux and ``MBody1_Release.exe`` on Windows.





.. _doxid-d7/d98/Quickstart_1ownmodel:

Defining a New Model in GeNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

According to the work flow outlined above, there are several steps to be completed to define a neuronal network model.

#. The neuronal network of interest is defined in a model definition file, e.g. ``Example1.cc``.

#. Within the the model definition file ``Example1.cc``, the following tasks need to be completed:
   
   a) The GeNN file ``modelSpec.h`` needs to be included,
   
   .. ref-code-block:: cpp
   
   	#include "modelSpec.h"
   
   b) The values for initial variables and parameters for neuron and synapse populations need to be defined, e.g.
   
   .. ref-code-block:: cpp
   
   	:ref:`NeuronModels::PoissonNew::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` poissonParams(
   	  10.0);      // 0 - firing rate
   
   would define the (homogeneous) parameters for a population of Poisson neurons. The number of required parameters and their meaning is defined by the neuron or synapse type. Refer to the :ref:`User Manual <doxid-dc/d05/UserManual>` for details. We recommend, however, to use comments like in the above example to achieve maximal clarity of each parameter's meaning.
   
   If heterogeneous parameter values are required for a particular population of neurons (or synapses), they need to be defined as "variables" rather than parameters. See the :ref:`User Manual <doxid-dc/d05/UserManual>` for how to define new neuron (or synapse) types and the :ref:`Defining a new variable initialisation snippet <doxid-d4/dc6/sectVariableInitialisation_1sect_new_var_init>` section for more information on initialising these variables to hetererogenous values.
   
   c) The actual network needs to be defined in the form of a function ``modelDefinition``, i.e.
   
   .. ref-code-block:: cpp
   
   	void modelDefinition(:ref:`ModelSpec <doxid-da/dfd/classModelSpec>` &model);
   
   The name ``modelDefinition`` and its parameter of type ``:ref:`ModelSpec <doxid-da/dfd/classModelSpec>`&`` are fixed and cannot be changed if GeNN is to recognize it as a model definition.
   
   d) Inside modelDefinition(), The time step ``DT`` needs to be defined, e.g.
   
   .. ref-code-block:: cpp
   
   	model.setDT(0.1);
   
   All provided examples and pre-defined model elements in GeNN work with units of mV, ms, nF and muS. However, the choice of units is entirely left to the user if custom model elements are used.
   
   ``MBody1.cc`` shows a typical example of a model definition function. In its core it contains calls to :ref:`ModelSpec::addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>` and :ref:`ModelSpec::addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>` to build up the network. For a full range of options for defining a network, refer to the :ref:`User Manual <doxid-dc/d05/UserManual>`.

#. The programmer defines their own "user-side" modeling code similar to the code in ``userproject/MBody1_project/model/MBody1Sim.cc``. In this code,
   
   a) They manually define the connectivity matrices between neuron groups. Refer to the :ref:`Synaptic matrix types <doxid-d5/d39/subsect34>` section for the required format of connectivity matrices for dense or sparse connectivities.
   
   b) They define input patterns (e.g. for Poisson neurons like in the MBody1 example) or individual initial values for neuron and / or synapse variables. The initial values given in the ``modelDefinition`` are automatically applied homogeneously to every individual neuron or synapse in each of the neuron or synapse groups.
   
   c) They use ``stepTime()`` to run one time step on either the CPU or GPU depending on the options passed to genn-buildmodel.
   
   d) They use functions like ``copyStateFromDevice()`` etc to transfer the results from GPU calculations to the main memory of the host computer for further processing.
   
   e) They analyze the results. In the most simple case this could just be writing the relevant data to output files.

:ref:`Previous <doxid-d8/d99/Installation>` \| :ref:`Top <doxid-d7/d98/Quickstart>` \| :ref:`Next <doxid-d9/d61/Examples>`

