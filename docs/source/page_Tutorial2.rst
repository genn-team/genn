.. index:: pair: page; Tutorial 2
.. _doxid-dc/d7e/Tutorial2:

Tutorial 2
==========

In this tutorial we will learn to add synapsePopulations to connect neurons in neuron groups to each other with synaptic models. As an example we will connect the ten Hodgkin-Huxley neurons from tutorial 1 in a ring of excitatory synapses.

First, copy the files from Tutorial 1 into a new directory and rename the ``tenHHModel.cc`` to ``tenHHRingModel.cc`` and ``tenHHSimulation.cc`` to ``tenHHRingSimulation.cc``, e.g. on Linux or Mac:

.. ref-code-block:: cpp

	>> cp -r tenHH_project tenHHRing_project
	>> cd tenHHRing_project
	>> mv tenHHModel.cc tenHHRingModel.cc
	>> mv tenHHSimulation.cc tenHHRingSimulation.cc

Finally, to reduce confusion we should rename the model itself. Open ``tenHHRingModel.cc``, change the model name inside,

.. ref-code-block:: cpp

	model.:ref:`setName <doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4>`("tenHHRing");



.. _doxid-dc/d7e/Tutorial2_1SynapseMatrix:

Defining the Detailed Synaptic Connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We want to connect our ten neurons into a ring where each neuron connects to its neighbours. In order to initialise this connectivity we need to add a sparse connectivity initialisation snippet at the top of ``tenHHRingModel.cc`` :

.. ref-code-block:: cpp

	class Ring : public :ref:`InitSparseConnectivitySnippet::Base <doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`
	{
	public:
	    :ref:`DECLARE_SNIPPET <doxid-de/d6c/snippet_8h_1ac5727a6720d28f034afadde948ed6e9a>`(Ring, 0);
	    :ref:`SET_ROW_BUILD_CODE <doxid-de/d51/initSparseConnectivitySnippet_8h_1a3758f6bc5bc997383426d5f277b8acc9>`(
	        "$(addSynapse, ($(id_pre) + 1) % $(num_post));\n"
	        "$(endRow);\n");
	    :ref:`SET_MAX_ROW_LENGTH <doxid-de/d51/initSparseConnectivitySnippet_8h_1a338915170111c85ba647e848d28ee2a9>`(1);
	};
	:ref:`IMPLEMENT_SNIPPET <doxid-de/d6c/snippet_8h_1af3c47debe5fc34060e716d7db25462ab>`(Ring);

The ``SET_ROW_BUILD_CODE`` code string will be called to generate each row of the synaptic matrix (connections coming from a single presynaptic neuron) and, in this case, each row consists of a single synapses from the presynaptic neuron $(id_pre) to $(id_pre) + 1 (the modulus operator is used to ensure that the final connection between neuron ``9`` and ``0`` is made correctly). In order to allow GeNN to better optimise the generated code we also provide a maximum row length. In this case each row always contains only one synapse but, when more complex connectivity is used, the number of neurons in the pre and postsynaptic population as well as any parameters used to configure the snippet can be accessed from this function. When defining GeNN code strings, the $(VariableName) syntax is used to refer to variables provided by GeNN and the $(FunctionName, Parameter1,...) syntax is used to call functions provided by GeNN.





.. _doxid-dc/d7e/Tutorial2_1addSynapse:

Adding Synaptic connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we need additional initial values and parameters for the synapse and post-synaptic models. We will use the standard :ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>` weight update model and :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>` post-synaptic model. They need the following initial variables and parameters:

.. ref-code-block:: cpp

	WeightUpdateModels::StaticPulse::VarValues s_ini(
	    -0.2); // 0 - g: the synaptic conductance value
	
	:ref:`PostsynapticModels::ExpCond::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` ps_p(
	    1.0,    // 0 - tau_S: decay time constant for S [ms]
	    -80.0); // 1 - Erev: Reversal potential

the :ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>` weight update model has no parameters and the :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>` post-synaptic model has no state variables.

We can then add a synapse population at the end of the ``modelDefinition(...)`` function,

.. ref-code-block:: cpp

	model.:ref:`addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`<:ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>`, :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>`>(
	    "Pop1self", :ref:`SynapseMatrixType::SPARSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>`, 10,
	    "Pop1", "Pop1",
	    {}, s_ini,
	    ps_p, {},
	    initConnectivity<Ring>());

The addSynapsePopulation parameters are

* WeightUpdateModel: template parameter specifying the type of weight update model (derived from :ref:`WeightUpdateModels::Base <doxid-d2/d05/classWeightUpdateModels_1_1Base>`).

* PostsynapticModel: template parameter specifying the type of postsynaptic model (derived from :ref:`PostsynapticModels::Base <doxid-d1/d3a/classPostsynapticModels_1_1Base>`).

* name string containing unique name of synapse population.

* mtype how the synaptic matrix associated with this synapse population should be represented. Here :ref:`SynapseMatrixType::SPARSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>` means that there will be sparse connectivity and each connection will have the same weight (-0.2 as specified previously).

* delayStep integer specifying number of timesteps of propagation delay that spikes travelling through this synapses population should incur (or NO_DELAY for none)

* src string specifying name of presynaptic (source) population

* trg string specifying name of postsynaptic (target) population

* weightParamValues parameters for weight update model wrapped in WeightUpdateModel::ParamValues object.

* weightVarInitialisers initial values or initialisation snippets for the weight update model's state variables wrapped in a WeightUpdateModel::VarValues object.

* postsynapticParamValues parameters for postsynaptic model wrapped in PostsynapticModel::ParamValues object.

* postsynapticVarInitialisers initial values or initialisation snippets for the postsynaptic model wrapped in PostsynapticModel::VarValues object.

* connectivityInitialiser snippet and any paramaters (in this case there are none) used to initialise the synapse population's sparse connectivity.

Adding the addSynapsePopulation command to the model definition informs GeNN that there will be synapses between the named neuron populations, here between population ``Pop1`` and itself. At this point our model definition file ``tenHHRingModel.cc`` should look like this

.. ref-code-block:: cpp

	// Model definition file tenHHRing.cc
	#include "modelSpec.h"
	
	class Ring : public :ref:`InitSparseConnectivitySnippet::Base <doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`
	{
	public:
	    :ref:`DECLARE_SNIPPET <doxid-de/d6c/snippet_8h_1ac5727a6720d28f034afadde948ed6e9a>`(Ring, 0);
	    :ref:`SET_ROW_BUILD_CODE <doxid-de/d51/initSparseConnectivitySnippet_8h_1a3758f6bc5bc997383426d5f277b8acc9>`(
	        "$(addSynapse, ($(id_pre) + 1) % $(num_post));\n"
	        "$(endRow);\n");
	    :ref:`SET_MAX_ROW_LENGTH <doxid-de/d51/initSparseConnectivitySnippet_8h_1a338915170111c85ba647e848d28ee2a9>`(1);
	};
	:ref:`IMPLEMENT_SNIPPET <doxid-de/d6c/snippet_8h_1af3c47debe5fc34060e716d7db25462ab>`(Ring);
	
	void modelDefinition(:ref:`ModelSpec <doxid-da/dfd/classModelSpec>` &model)
	{
	    // definition of tenHHRing
	    model.:ref:`setDT <doxid-da/dfd/classModelSpec_1a329236a3b07044b82bfda5b4f741d8e1>`(0.1);
	    model.:ref:`setName <doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4>`("tenHHRing");
	
	    :ref:`NeuronModels::TraubMiles::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` p(
	        7.15,       // 0 - gNa: Na conductance in muS
	        50.0,       // 1 - ENa: Na equi potential in mV
	        1.43,       // 2 - gK: K conductance in muS
	        -95.0,      // 3 - EK: K equi potential in mV
	        0.02672,    // 4 - gl: leak conductance in muS
	        -63.563,    // 5 - El: leak equi potential in mV
	        0.143);     // 6 - Cmem: membr. capacity density in nF
	
	    :ref:`NeuronModels::TraubMiles::VarValues <doxid-d6/d24/classModels_1_1VarInitContainerBase>` ini(
	        -60.0,         // 0 - membrane potential V
	        0.0529324,     // 1 - prob. for Na channel activation m
	        0.3176767,     // 2 - prob. for not Na channel blocking h
	        0.5961207);    // 3 - prob. for K channel activation n
	
	    model.:ref:`addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>`<:ref:`NeuronModels::TraubMiles <doxid-d2/dc3/classNeuronModels_1_1TraubMiles>`>("Pop1", 10, p, ini);
	
	    WeightUpdateModels::StaticPulse::VarValues s_ini(
	         -0.2); // 0 - g: the synaptic conductance value
	
	    :ref:`PostsynapticModels::ExpCond::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` ps_p(
	        1.0,    // 0 - tau_S: decay time constant for S [ms]
	        -80.0); // 1 - Erev: Reversal potential
	
	    model.:ref:`addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`<:ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>`, :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>`>(
	        "Pop1self", :ref:`SynapseMatrixType::SPARSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>`, 100,
	        "Pop1", "Pop1",
	        {}, s_ini,
	        ps_p, {},
	        initConnectivity<Ring>());
	}

We can now build our new model:

.. ref-code-block:: cpp

	>> genn-buildmodel.sh tenHHRingModel.cc

Again, if you don't have an NVIDIA GPU and are running GeNN in CPU_ONLY mode, you can instead build with the ``-c`` option as described in :ref:`Tutorial 1 <doxid-d5/dbb/Tutorial1>`.

Now we can open the ``tenHHRingSimulation.cc`` file and update the file name of the model includes to match the name we set previously:

.. ref-code-block:: cpp

	// tenHHRingModel simulation code
	#include "tenHHRing_CODE/definitions.h"

Additionally, we need to add a call to a second initialisation function to ``main()`` after we call ``initialize()`` :

.. ref-code-block:: cpp

	initializeSparse();

This initializes any variables associated with the sparse connectivity we have added (and will also copy any manually initialised variables to the GPU). Then, after using the ``genn-create-user-project`` tool to create a new project with a model name of ``tenHHRing`` and using ``tenHHRingSimulation.cc`` rather than ``tenHHSimulation.cc``, we can build and run our new simulator in the same way we did in :ref:`Tutorial 1 <doxid-d5/dbb/Tutorial1>`. However, even after all our hard work, if we plot the content of the first column against the subsequent 10 columns of ``tenHHexample.V.dat`` it looks very similar to the plot we obtained at the end of :ref:`Tutorial 1 <doxid-d5/dbb/Tutorial1>`.

.. image:: tenHHRingexample1.png



.. image:: tenHHRingexample1.png
	:alt: width=10cm

This is because none of the neurons are spiking so there are no spikes to propagate around the ring.





.. _doxid-dc/d7e/Tutorial2_1initialConditions:

Providing initial stimuli
~~~~~~~~~~~~~~~~~~~~~~~~~

We can use a :ref:`NeuronModels::SpikeSource <doxid-d5/d1f/classNeuronModels_1_1SpikeSource>` to inject an initial spike into the first neuron in the ring during the first timestep to start spikes propagating. Firstly we need to define another sparse connectivity initialisation snippet at the top of ``tenHHRingModel.cc`` which simply creates a single synapse on the first row of the synaptic matrix:

.. ref-code-block:: cpp

	class FirstToFirst : public :ref:`InitSparseConnectivitySnippet::Base <doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`
	{
	public:
	    :ref:`DECLARE_SNIPPET <doxid-de/d6c/snippet_8h_1ac5727a6720d28f034afadde948ed6e9a>`(FirstToFirst, 0);
	    :ref:`SET_ROW_BUILD_CODE <doxid-de/d51/initSparseConnectivitySnippet_8h_1a3758f6bc5bc997383426d5f277b8acc9>`(
	        "if($(id_pre) == 0) {\n"
	        "   $(addSynapse, $(id_pre));\n"
	        "}\n"
	        "$(endRow);\n");
	    :ref:`SET_MAX_ROW_LENGTH <doxid-de/d51/initSparseConnectivitySnippet_8h_1a338915170111c85ba647e848d28ee2a9>`(1);
	};
	:ref:`IMPLEMENT_SNIPPET <doxid-de/d6c/snippet_8h_1af3c47debe5fc34060e716d7db25462ab>`(FirstToFirst);

We then need to add it to the network by adding the following to the end of the ``modelDefinition(...)`` function:

.. ref-code-block:: cpp

	model.:ref:`addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>`<:ref:`NeuronModels::SpikeSource <doxid-d5/d1f/classNeuronModels_1_1SpikeSource>`>("Stim", 1, {}, {});
	model.:ref:`addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`<:ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>`, :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>`>(
	    "StimPop1", :ref:`SynapseMatrixType::SPARSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>`, :ref:`NO_DELAY <doxid-dc/de1/modelSpec_8h_1a291aa33d0e485ee09a6881cf8056e13c>`,
	    "Stim", "Pop1",
	    {}, s_ini,
	    ps_p, {},
	    initConnectivity<FirstToFirst>());

and finally inject a spike in the first timestep (in the same way that the ``t`` variable is provided by GeNN to keep track of the current simulation time in milliseconds, ``iT`` is provided to keep track of it in timesteps):

.. ref-code-block:: cpp

	if(iT == 0) {
	    spikeCount_Stim = 1;
	    spike_Stim[0] = 0;
	    pushStimCurrentSpikesToDevice();
	}

``spike_Stim[n]`` is used to specify the indices of the neurons in population ``Stim`` spikes which should emit spikes where :math:`n \in [0, \mbox{spikeCount\_Stim} )`.

At this point our user code ``tenHHRingModel.cc`` should look like this

.. ref-code-block:: cpp

	// Model definintion file tenHHRing.cc
	#include "modelSpec.h"
	
	class Ring : public :ref:`InitSparseConnectivitySnippet::Base <doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`
	{
	public:
	    :ref:`DECLARE_SNIPPET <doxid-de/d6c/snippet_8h_1ac5727a6720d28f034afadde948ed6e9a>`(Ring, 0);
	    :ref:`SET_ROW_BUILD_CODE <doxid-de/d51/initSparseConnectivitySnippet_8h_1a3758f6bc5bc997383426d5f277b8acc9>`(
	        "$(addSynapse, ($(id_pre) + 1) % $(num_post));\n"
	        "$(endRow);\n");
	    :ref:`SET_MAX_ROW_LENGTH <doxid-de/d51/initSparseConnectivitySnippet_8h_1a338915170111c85ba647e848d28ee2a9>`(1);
	};
	:ref:`IMPLEMENT_SNIPPET <doxid-de/d6c/snippet_8h_1af3c47debe5fc34060e716d7db25462ab>`(Ring);
	
	class FirstToFirst : public :ref:`InitSparseConnectivitySnippet::Base <doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`
	{
	public:
	    :ref:`DECLARE_SNIPPET <doxid-de/d6c/snippet_8h_1ac5727a6720d28f034afadde948ed6e9a>`(FirstToFirst, 0);
	    :ref:`SET_ROW_BUILD_CODE <doxid-de/d51/initSparseConnectivitySnippet_8h_1a3758f6bc5bc997383426d5f277b8acc9>`(
	        "if($(id_pre) == 0) {\n"
	        "   $(addSynapse, $(id_pre));\n"
	        "}\n"
	        "$(endRow);\n");
	    :ref:`SET_MAX_ROW_LENGTH <doxid-de/d51/initSparseConnectivitySnippet_8h_1a338915170111c85ba647e848d28ee2a9>`(1);
	};
	:ref:`IMPLEMENT_SNIPPET <doxid-de/d6c/snippet_8h_1af3c47debe5fc34060e716d7db25462ab>`(FirstToFirst);
	
	void modelDefinition(:ref:`ModelSpec <doxid-da/dfd/classModelSpec>` &model)
	{
	    // definition of tenHHRing
	    model.:ref:`setDT <doxid-da/dfd/classModelSpec_1a329236a3b07044b82bfda5b4f741d8e1>`(0.1);
	    model.:ref:`setName <doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4>`("tenHHRing");
	
	    :ref:`NeuronModels::TraubMiles::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` p(
	        7.15,       // 0 - gNa: Na conductance in muS
	        50.0,       // 1 - ENa: Na equi potential in mV
	        1.43,       // 2 - gK: K conductance in muS
	        -95.0,      // 3 - EK: K equi potential in mV
	        0.02672,    // 4 - gl: leak conductance in muS
	        -63.563,    // 5 - El: leak equi potential in mV
	        0.143);     // 6 - Cmem: membr. capacity density in nF
	
	    :ref:`NeuronModels::TraubMiles::VarValues <doxid-d6/d24/classModels_1_1VarInitContainerBase>` ini(
	        -60.0,         // 0 - membrane potential V
	        0.0529324,     // 1 - prob. for Na channel activation m
	        0.3176767,     // 2 - prob. for not Na channel blocking h
	        0.5961207);    // 3 - prob. for K channel activation n
	
	    model.:ref:`addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>`<:ref:`NeuronModels::TraubMiles <doxid-d2/dc3/classNeuronModels_1_1TraubMiles>`>("Pop1", 10, p, ini);
	    model.:ref:`addNeuronPopulation <doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>`<:ref:`NeuronModels::SpikeSource <doxid-d5/d1f/classNeuronModels_1_1SpikeSource>`>("Stim", 1, {}, {});
	
	    WeightUpdateModels::StaticPulse::VarValues s_ini(
	         -0.2); // 0 - g: the synaptic conductance value
	
	    :ref:`PostsynapticModels::ExpCond::ParamValues <doxid-da/d76/classSnippet_1_1ValueBase>` ps_p(
	        1.0,    // 0 - tau_S: decay time constant for S [ms]
	        -80.0); // 1 - Erev: Reversal potential
	
	    model.:ref:`addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`<:ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>`, :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>`>(
	        "Pop1self", :ref:`SynapseMatrixType::SPARSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>`, 100,
	        "Pop1", "Pop1",
	        {}, s_ini,
	        ps_p, {},
	        initConnectivity<Ring>());
	
	    model.:ref:`addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`<:ref:`WeightUpdateModels::StaticPulse <doxid-d9/d74/classWeightUpdateModels_1_1StaticPulse>`, :ref:`PostsynapticModels::ExpCond <doxid-d5/d27/classPostsynapticModels_1_1ExpCond>`>(
	        "StimPop1", :ref:`SynapseMatrixType::SPARSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>`, :ref:`NO_DELAY <doxid-dc/de1/modelSpec_8h_1a291aa33d0e485ee09a6881cf8056e13c>`,
	        "Stim", "Pop1",
	        {}, s_ini,
	        ps_p, {},
	        initConnectivity<FirstToFirst>());
	}

and ``tenHHRingSimulation.cc`` ` should look like this:

.. ref-code-block:: cpp

	// Standard C++ includes
	#include <fstream>
	
	// tenHHRing simulation code
	#include "tenHHRing_CODE/definitions.h"
	
	int main()
	{
	    allocateMem();
	    initialize();
	    initializeSparse();
	
	    std::ofstream os("tenHHRing_output.V.dat");
	    while(t < 200.0f) {
	        if(iT == 0) {
	            glbSpkStim[0] = 0;
	            glbSpkCntStim[0] = 1;
	            pushStimCurrentSpikesToDevice();
	        }
	
	        stepTimeU();
	        pullVPop1FromDevice();
	
	        os << t << " ";
	        for (int j= 0; j < 10; j++) {
	            os << VPop1[j] << " ";
	        }
	        os << std::endl;
	    }
	    os.close();
	    return 0;
	}

Finally if we build, make and run this model; and plot the first 200 ms of the ten neurons' membrane voltages - they now looks like this:

.. image:: tenHHRingexample2.png



.. image:: tenHHRingexample2.png
	:alt: width=10cm

:ref:`Previous <doxid-d5/dbb/Tutorial1>` \| :ref:`Top <doxid-dc/d7e/Tutorial2>` \| :ref:`Next <doxid-d0/da6/UserGuide>`

