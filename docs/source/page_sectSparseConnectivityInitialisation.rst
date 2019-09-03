.. index:: pair: page; Sparse connectivity initialisation
.. _doxid-d5/dd4/sectSparseConnectivityInitialisation:

Sparse connectivity initialisation
==================================

Synaptic connectivity implemented using :ref:`SynapseMatrixConnectivity::SPARSE <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0459833ba9cad7cfd7bbfe10d7bbbe6e>` and :ref:`SynapseMatrixConnectivity::BITMASK <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0287e103671bf22378919a64d4b70699>` can be automatically initialised.

This can be done using one of a number of predefined *sparse connectivity initialisation snippets* :

* :ref:`InitSparseConnectivitySnippet::OneToOne <doxid-d5/dd3/classInitSparseConnectivitySnippet_1_1OneToOne>`

* :ref:`InitSparseConnectivitySnippet::FixedProbability <doxid-df/d2e/classInitSparseConnectivitySnippet_1_1FixedProbability>`

* :ref:`InitSparseConnectivitySnippet::FixedProbabilityNoAutapse <doxid-d2/d17/classInitSparseConnectivitySnippet_1_1FixedProbabilityNoAutapse>`

For example, to initialise synaptic connectivity with a 10% connection probability (allowing connections between neurons with the same id):

.. ref-code-block:: cpp

	InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(0.1);
	    
	model.:ref:`addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`<...>(
	        ...
	        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));



.. _doxid-d5/dd4/sectSparseConnectivityInitialisation_1sect_new_sparse_connect:

Defining a new sparse connectivity snippet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly to variable initialisation snippets, sparse connectivity initialisation snippets can be created by simply defining a class in the model description.

For example, the following sparse connectivity initialisation snippet could be used to initialise a 'ring' of connectivity where each neuron is connected to a number of subsequent neurons specified using the ``numNeighbours`` parameter:

.. ref-code-block:: cpp

	class Ring : public :ref:`InitSparseConnectivitySnippet::Base <doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`
	{
	public:
	    :ref:`DECLARE_SNIPPET <doxid-de/d6c/snippet_8h_1ac5727a6720d28f034afadde948ed6e9a>`(Ring, 1);
	
	    :ref:`SET_ROW_BUILD_STATE_VARS <doxid-de/d51/initSparseConnectivitySnippet_8h_1abfe3722618884af89eb9c64e1345c03f>`({{"offset", {"unsigned int", 1}}}});
	    :ref:`SET_ROW_BUILD_CODE <doxid-de/d51/initSparseConnectivitySnippet_8h_1a3758f6bc5bc997383426d5f277b8acc9>`(
	        "const unsigned int target = ($(id_pre) + offset) % $(num_post);\n"
	        "$(addSynapse, target);\n"
	        "offset++;\n"
	        "if(offset > (unsigned int)$(numNeighbours)) {\n"
	        "   $(endRow);\n"
	        "}\n");
	
	    :ref:`SET_PARAM_NAMES <doxid-de/d6c/snippet_8h_1a75315265035fd71c5b5f7d7f449edbd7>`({"numNeighbours"});
	    :ref:`SET_CALC_MAX_ROW_LENGTH_FUNC <doxid-de/d51/initSparseConnectivitySnippet_8h_1adc763f727358b11685ddeab7ca8434f2>`(
	        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
	        {
	            return (unsigned int)pars[0];
	        });
	    :ref:`SET_CALC_MAX_COL_LENGTH_FUNC <doxid-de/d51/initSparseConnectivitySnippet_8h_1ad59a50b968b2b9dc03093ea1306eec40>`(
	        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
	        {
	            return (unsigned int)pars[0];
	        });
	};
	:ref:`IMPLEMENT_SNIPPET <doxid-de/d6c/snippet_8h_1af3c47debe5fc34060e716d7db25462ab>`(Ring);

Each *row* of sparse connectivity is initialised independantly by running the snippet of code specified using the ``:ref:`SET_ROW_BUILD_CODE() <doxid-de/d51/initSparseConnectivitySnippet_8h_1a3758f6bc5bc997383426d5f277b8acc9>``` macro within a loop. The $(num_post) variable can be used to access the number of neurons in the postsynaptic population and the $(id_pre) variable can be used to access the index of the presynaptic neuron associated with the row being generated. The ``:ref:`SET_ROW_BUILD_STATE_VARS() <doxid-de/d51/initSparseConnectivitySnippet_8h_1abfe3722618884af89eb9c64e1345c03f>``` macro can be used to initialise state variables outside of the loop - in this case ``offset`` which is used to count the number of synapses created in each row. Synapses are added to the row using the $(addSynapse, target) function and iteration is stopped using the $(endRow) function. To avoid having to manually call :ref:`SynapseGroup::setMaxConnections <doxid-dc/dfa/classSynapseGroup_1aab6b2fb0ad30189bc11ee3dd7d48dbb2>` and :ref:`SynapseGroup::setMaxSourceConnections <doxid-dc/dfa/classSynapseGroup_1a93b12c08d634f1a2300f1b91ef34ea24>`, sparse connectivity snippets can also provide code to calculate the maximum row and column lengths this connectivity will result in using the :ref:`SET_CALC_MAX_ROW_LENGTH_FUNC() <doxid-de/d51/initSparseConnectivitySnippet_8h_1adc763f727358b11685ddeab7ca8434f2>` and :ref:`SET_CALC_MAX_COL_LENGTH_FUNC() <doxid-de/d51/initSparseConnectivitySnippet_8h_1ad59a50b968b2b9dc03093ea1306eec40>` macros. Alternatively, if the maximum row or column length is constant, the ``:ref:`SET_MAX_ROW_LENGTH() <doxid-de/d51/initSparseConnectivitySnippet_8h_1a338915170111c85ba647e848d28ee2a9>``` and ``:ref:`SET_MAX_COL_LENGTH() <doxid-de/d51/initSparseConnectivitySnippet_8h_1a9d72764eb9a910bba6d4a1776717ba02>``` shorthand macros can be used.





.. _doxid-d5/dd4/sectSparseConnectivityInitialisation_1sect_sparse_connect_init_modes:

Sparse connectivity locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have defined **how** sparse connectivity is going to be initialised, similarly to variables, you can control **where** it is allocated. This is controlled using the same ``VarLocations`` options described in section :ref:`Variable locations <doxid-d4/dc6/sectVariableInitialisation_1sect_var_init_modes>` and can either be set using the model default specifiued with ``:ref:`ModelSpec::setDefaultSparseConnectivityLocation <doxid-da/dfd/classModelSpec_1a9bc61e7c5dce757de3a9b7479852ca72>``` or on a per-synapse group basis using ``:ref:`SynapseGroup::setSparseConnectivityLocation <doxid-dc/dfa/classSynapseGroup_1ae30487a9c1dc728cce45130821766fc8>```.

:ref:`Previous <doxid-d4/dc6/sectVariableInitialisation>` \| :ref:`Top <doxid-dc/d05/UserManual>` \| :ref:`Next <doxid-d5/dbb/Tutorial1>`

