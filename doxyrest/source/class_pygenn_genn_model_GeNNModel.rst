.. index:: pair: class; pygenn::genn_model::GeNNModel
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel:

class pygenn::genn_model::GeNNModel
===================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`GeNNModel <doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>` class This class helps to define, build and run a GeNN model from python. :ref:`More...<details-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	class GeNNModel: public object
	{
	public:
		// fields
	
		 :target:`use_backend<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab652ad87ca9dcdef1991307bf907be53>`;
		 :target:`default_var_location<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab76469f3e0329c1bc7827b6b98d276b3>`;
		 :target:`model_name<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab1164b8374a4b390e77bc32effed638c>`;
		 :target:`neuron_populations<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a64509d979db91e40ebb1c92e0e246e94>`;
		 :target:`synapse_populations<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a97cca022c95e6744c4b0f6ea8f770565>`;
		 :target:`current_sources<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a93daad6952a3dd9e66bc292617ae1830>`;
		 :target:`dT<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afa2abcf499adbe2380463cc68ec614db>`;
		 :target:`T<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1aa448edf34424822972747029870f73c7>`;

		// methods
	
		def :ref:`__init__<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a97af74b8965a23e3ff5ccf76997e07c5>`(
			self self,
			precision precision = None,
			:ref:`model_name<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab1164b8374a4b390e77bc32effed638c>` model_name = "GeNNModel",
			enable_debug enable_debug = False,
			backend backend = None
			);
	
		def :target:`use_backend<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a4735ba33ebee63fe36e2da3f384ead4a>`(self self);
	
		def :target:`use_backend<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a3137503c327f26fcc6d7323505233745>`(
			self self,
			backend backend
			);
	
		def :ref:`default_var_location<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1abac6e37ea0e6fc1a3b092543a9012951>`(self self);
	
		def :target:`default_var_location<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a6fb04c026cd6daf493fb160b939d81a1>`(
			self self,
			location location
			);
	
		def :ref:`default_sparse_connectivity_location<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a08c1d0a533cc7b2a6bb41e421d57b888>`(location location);
	
		def :target:`default_sparse_connectivity_location<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a623389c1d66d61f0ccce2870c2a3b946>`(
			self self,
			location location
			);
	
		def :ref:`model_name<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a657e703ee01779146f1ccb8351ede55b>`(self self);
	
		def :target:`model_name<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a4e21fa1268613abb29d4d85de6084638>`(
			self self,
			model_name model_name
			);
	
		def :ref:`t<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a48193c41cf24f25cdd2970cab7c88e87>`(self self);
	
		def :target:`t<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a290cab59495dd92ea26f6d379ba3c373>`(
			self self,
			t t
			);
	
		def :ref:`timestep<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1aef0498990a471f39daad3fd841df7111>`(self self);
	
		def :target:`timestep<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a8e25caa5f6fa75b1034c15d71ec86e59>`(
			self self,
			timestep timestep
			);
	
		def :ref:`dT<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ac249821001e0f49753b17ad46c68e049>`(self self);
	
		def :target:`dT<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a487dbad5a2661672d5fea901f1eb71f3>`(
			self self,
			dt dt
			);
	
		def :ref:`add_neuron_population<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a100324e214546094fced5e66fb0582d4>`(
			self self,
			pop_name pop_name,
			num_neurons num_neurons,
			neuron neuron,
			param_space param_space,
			var_space var_space
			);
	
		def :ref:`add_synapse_population<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab573f92c82ec25d5c80f94070b6008c0>`(
			self self,
			pop_name pop_name,
			matrix_type matrix_type,
			delay_steps delay_steps,
			source source,
			target target,
			w_update_model w_update_model,
			wu_param_space wu_param_space,
			wu_var_space wu_var_space,
			wu_pre_var_space wu_pre_var_space,
			wu_post_var_space wu_post_var_space,
			postsyn_model postsyn_model,
			ps_param_space ps_param_space,
			ps_var_space ps_var_space,
			connectivity_initialiser connectivity_initialiser = None
			);
	
		def :ref:`add_current_source<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a20b03cbc0200038e8dd9950553fe5152>`(
			self self,
			cs_name cs_name,
			current_source_model current_source_model,
			pop_name pop_name,
			param_space param_space,
			var_space var_space
			);
	
		def :ref:`build<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a481795d2e7a45410799786a1c5122834>`(self self, path_to_model path_to_model = "./");
		def :ref:`load<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a3f5a720288c67d0a87a1a270c7ff2f2c>`(self self);
		def :ref:`reinitialise<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a475a22c65205debb202eb98d4dc929f2>`(self self);
		def :target:`step_time<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a6d71c8d91805b64c5a69fc62ea3dfbdb>`(self self);
		def :ref:`pull_state_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a6c5c3ab36b8348eade67585325652a70>`(self self, pop_name pop_name);
		def :ref:`pull_spikes_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a866e2b2d9f823c67365bce6395b82387>`(self self, pop_name pop_name);
		def :ref:`pull_current_spikes_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a3ea2a76a16f5aa01ac244d3a4eb289f4>`(self self, pop_name pop_name);
		def :ref:`pull_connectivity_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a915100ccf83beff4923c503475aad46a>`(self self, pop_name pop_name);
		def :ref:`pull_var_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a9a24190a85b0e01e1f22e030a4a6a42b>`(self self, pop_name pop_name, var_name var_name);
		def :ref:`push_state_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afc6a5ef4b76997365450b40eaca00db0>`(self self, pop_name pop_name);
		def :ref:`push_spikes_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a2c68d702b36428da03126c0dd13d4c9e>`(self self, pop_name pop_name);
		def :ref:`push_current_spikes_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a02586b0bf4c6969e7013778ced108194>`(self self, pop_name pop_name);
		def :ref:`push_connectivity_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a88a9e18417e48815a2534a2e6dead826>`(self self, pop_name pop_name);
		def :ref:`push_var_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a8217fc38ecb1bbbd249bfee27455ac90>`(self self, pop_name pop_name, var_name var_name);
		def :ref:`end<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a06269039e7798cafe00f637e5df88410>`(self self);
	};
.. _details-db/d57/classpygenn_1_1genn__model_1_1GeNNModel:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`GeNNModel <doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>` class This class helps to define, build and run a GeNN model from python.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a97af74b8965a23e3ff5ccf76997e07c5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__(
		self self,
		precision precision = None,
		:ref:`model_name<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab1164b8374a4b390e77bc32effed638c>` model_name = "GeNNModel",
		enable_debug enable_debug = False,
		backend backend = None
		)

Init :ref:`GeNNModel <doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- precision

		- string precision as string ("float", "double" or "long double"). defaults to float.

	*
		- model_name

		- string name of the model. Defaults to "GeNNModel".

	*
		- enable_debug

		- boolean enable debug mode. Disabled by default.

	*
		- backend

		- string specifying name of backend module to use Defaults to None to pick 'best' backend for your system

.. index:: pair: function; default_var_location
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1abac6e37ea0e6fc1a3b092543a9012951:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def default_var_location(self self)

Default variable location - defines where state variables are initialised.

.. index:: pair: function; default_sparse_connectivity_location
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a08c1d0a533cc7b2a6bb41e421d57b888:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def default_sparse_connectivity_location(location location)

Default sparse connectivity mode - where connectivity is initialised.

.. index:: pair: function; model_name
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a657e703ee01779146f1ccb8351ede55b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def model_name(self self)

Name of the model.

.. index:: pair: function; t
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a48193c41cf24f25cdd2970cab7c88e87:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def t(self self)

Simulation time in ms.

.. index:: pair: function; timestep
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1aef0498990a471f39daad3fd841df7111:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def timestep(self self)

Simulation time step.

.. index:: pair: function; dT
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ac249821001e0f49753b17ad46c68e049:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def dT(self self)

Step size.

.. index:: pair: function; add_neuron_population
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a100324e214546094fced5e66fb0582d4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_neuron_population(
		self self,
		pop_name pop_name,
		num_neurons num_neurons,
		neuron neuron,
		param_space param_space,
		var_space var_space
		)

Add a neuron population to the GeNN model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pop_name

		- name of the new population

	*
		- num_neurons

		- number of neurons in the new population

	*
		- neuron

		- type of the :ref:`NeuronModels <doxid-da/dac/namespaceNeuronModels>` class as string or instance of neuron class derived from ``pygenn.genn_wrapper.NeuronModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`)

	*
		- param_space

		- dict with param values for the :ref:`NeuronModels <doxid-da/dac/namespaceNeuronModels>` class

	*
		- var_space

		- dict with initial variable values for the :ref:`NeuronModels <doxid-da/dac/namespaceNeuronModels>` class

.. index:: pair: function; add_synapse_population
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab573f92c82ec25d5c80f94070b6008c0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_synapse_population(
		self self,
		pop_name pop_name,
		matrix_type matrix_type,
		delay_steps delay_steps,
		source source,
		target target,
		w_update_model w_update_model,
		wu_param_space wu_param_space,
		wu_var_space wu_var_space,
		wu_pre_var_space wu_pre_var_space,
		wu_post_var_space wu_post_var_space,
		postsyn_model postsyn_model,
		ps_param_space ps_param_space,
		ps_var_space ps_var_space,
		connectivity_initialiser connectivity_initialiser = None
		)

Add a synapse population to the GeNN model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pop_name

		- name of the new population

	*
		- matrix_type

		- type of the matrix as string

	*
		- delay_steps

		- delay in number of steps

	*
		- source

		- source neuron group

	*
		- target

		- target neuron group

	*
		- w_update_model

		- type of the :ref:`WeightUpdateModels <doxid-da/d80/namespaceWeightUpdateModels>` class as string or instance of weight update model class derived from ``pygenn.genn_wrapper.WeightUpdateModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`)

	*
		- wu_param_space

		- dict with param values for the :ref:`WeightUpdateModels <doxid-da/d80/namespaceWeightUpdateModels>` class

	*
		- wu_var_space

		- dict with initial values for :ref:`WeightUpdateModels <doxid-da/d80/namespaceWeightUpdateModels>` state variables

	*
		- wu_pre_var_space

		- dict with initial values for :ref:`WeightUpdateModels <doxid-da/d80/namespaceWeightUpdateModels>` presynaptic variables

	*
		- wu_post_var_space

		- dict with initial values for :ref:`WeightUpdateModels <doxid-da/d80/namespaceWeightUpdateModels>` postsynaptic variables

	*
		- postsyn_model

		- type of the :ref:`PostsynapticModels <doxid-db/dcb/namespacePostsynapticModels>` class as string or instance of postsynaptic model class derived from ``pygenn.genn_wrapper.PostsynapticModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611>`)

	*
		- ps_param_space

		- dict with param values for the :ref:`PostsynapticModels <doxid-db/dcb/namespacePostsynapticModels>` class

	*
		- ps_var_space

		- dict with initial variable values for the :ref:`PostsynapticModels <doxid-db/dcb/namespacePostsynapticModels>` class

	*
		- connectivity_initialiser

		- :ref:`InitSparseConnectivitySnippet::Init <doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>` for connectivity

.. index:: pair: function; add_current_source
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a20b03cbc0200038e8dd9950553fe5152:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_current_source(
		self self,
		cs_name cs_name,
		current_source_model current_source_model,
		pop_name pop_name,
		param_space param_space,
		var_space var_space
		)

Add a current source to the GeNN model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- cs_name

		- name of the new current source

	*
		- current_source_model

		- type of the :ref:`CurrentSourceModels <doxid-d6/dd3/namespaceCurrentSourceModels>` class as string or instance of :ref:`CurrentSourceModels <doxid-d6/dd3/namespaceCurrentSourceModels>` class derived from ``pygenn.genn_wrapper.CurrentSourceModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`)

	*
		- pop_name

		- name of the population into which the current source should be injected

	*
		- param_space

		- dict with param values for the :ref:`CurrentSourceModels <doxid-d6/dd3/namespaceCurrentSourceModels>` class

	*
		- var_space

		- dict with initial variable values for the :ref:`CurrentSourceModels <doxid-d6/dd3/namespaceCurrentSourceModels>` class

.. index:: pair: function; build
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a481795d2e7a45410799786a1c5122834:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def build(self self, path_to_model path_to_model = "./")

Finalize and build a GeNN model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- path_to_model

		- path where to place the generated model code. Defaults to the local directory.

.. index:: pair: function; load
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a3f5a720288c67d0a87a1a270c7ff2f2c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def load(self self)

import the model as shared library and initialize it

.. index:: pair: function; reinitialise
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a475a22c65205debb202eb98d4dc929f2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def reinitialise(self self)

reinitialise model to its original state without re-loading

.. index:: pair: function; pull_state_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a6c5c3ab36b8348eade67585325652a70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_state_from_device(self self, pop_name pop_name)

Pull state from the device for a given population.

.. index:: pair: function; pull_spikes_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a866e2b2d9f823c67365bce6395b82387:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_spikes_from_device(self self, pop_name pop_name)

Pull spikes from the device for a given population.

.. index:: pair: function; pull_current_spikes_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a3ea2a76a16f5aa01ac244d3a4eb289f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_current_spikes_from_device(self self, pop_name pop_name)

Pull spikes from the device for a given population.

.. index:: pair: function; pull_connectivity_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a915100ccf83beff4923c503475aad46a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_connectivity_from_device(self self, pop_name pop_name)

Pull connectivity from the device for a given population.

.. index:: pair: function; pull_var_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a9a24190a85b0e01e1f22e030a4a6a42b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_var_from_device(self self, pop_name pop_name, var_name var_name)

Pull variable from the device for a given population.

.. index:: pair: function; push_state_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afc6a5ef4b76997365450b40eaca00db0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_state_to_device(self self, pop_name pop_name)

Push state to the device for a given population.

.. index:: pair: function; push_spikes_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a2c68d702b36428da03126c0dd13d4c9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_spikes_to_device(self self, pop_name pop_name)

Push spikes to the device for a given population.

.. index:: pair: function; push_current_spikes_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a02586b0bf4c6969e7013778ced108194:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_current_spikes_to_device(self self, pop_name pop_name)

Push current spikes to the device for a given population.

.. index:: pair: function; push_connectivity_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a88a9e18417e48815a2534a2e6dead826:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_connectivity_to_device(self self, pop_name pop_name)

Push connectivity to the device for a given population.

.. index:: pair: function; push_var_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a8217fc38ecb1bbbd249bfee27455ac90:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_var_to_device(self self, pop_name pop_name, var_name var_name)

Push variable to the device for a given population.

.. index:: pair: function; end
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a06269039e7798cafe00f637e5df88410:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def end(self self)

Free memory.

