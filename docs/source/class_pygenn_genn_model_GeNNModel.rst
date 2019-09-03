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
	
		def :ref:`__init__<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab62bfb325a7b701e4763d2c2e27170da>`();
		def :target:`use_backend<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a69ec72251ff8334a0b8294cbdb049399>`();
		def :ref:`default_var_location<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a53bcaef460652366789cdeeee443d194>`();
		def :ref:`default_sparse_connectivity_location<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1abb13cf7f140b3a18d647f8644604d451>`();
		def :ref:`model_name<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afefdda5f27d55ca55d7e8e7703372785>`();
		def :ref:`t<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab67386f1c84f9c828827dfa54d609e78>`();
		def :ref:`timestep<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a204a3f0b579978cf3534d2c6dbb5c186>`();
		def :ref:`dT<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a076181e209188da2b3592afe8a3b04f1>`();
		def :ref:`add_neuron_population<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a6981e562d305fa486400451c9464e775>`();
		def :ref:`add_synapse_population<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1aa0d99a65e3ca5c179feb8c120e9da252>`();
		def :ref:`add_current_source<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ae3f4be499e6355cc917d0d3d6239815a>`();
		def :ref:`build<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a271d098822afa359ca392752b16683cc>`();
		def :ref:`load<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a8baf03355871cb66ea2368909433c22a>`();
		def :ref:`reinitialise<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a355a09159b9a00859300e01fae12687e>`();
		def :target:`step_time<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a37c76768715c6d4cc396ab73fb62a065>`();
		def :ref:`pull_state_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a71633dae5b0ac071b255857073bbf7e4>`();
		def :ref:`pull_spikes_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a4ba17d02864196b87f79cd25f9b326e7>`();
		def :ref:`pull_current_spikes_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1af88752774f48315be30964fb44226fb1>`();
		def :ref:`pull_connectivity_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a586c5aa3caf6bc232eab54c9ea7f3218>`();
		def :ref:`pull_var_from_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ad0fa3cd33bf965b706459d21328c49a9>`();
		def :ref:`push_state_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afe34bc9de940fc5ca5d23f02140caa1a>`();
		def :ref:`push_spikes_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a4109fc7d61910f2836b036e4ca80d002>`();
		def :ref:`push_current_spikes_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afbb2cb09e264df5615fc6e39d6fe5545>`();
		def :ref:`push_connectivity_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ac729fc62b67c260c0d91fb9c374bab62>`();
		def :ref:`push_var_to_device<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a912ec53be14d047d9a47e3d0fd1ba938>`();
		def :ref:`end<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1aa1ab378068b692100af0cec3014cdf5f>`();
	};
.. _details-db/d57/classpygenn_1_1genn__model_1_1GeNNModel:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`GeNNModel <doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>` class This class helps to define, build and run a GeNN model from python.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab62bfb325a7b701e4763d2c2e27170da:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__()

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
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a53bcaef460652366789cdeeee443d194:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def default_var_location()

Default variable location - defines where state variables are initialised.

.. index:: pair: function; default_sparse_connectivity_location
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1abb13cf7f140b3a18d647f8644604d451:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def default_sparse_connectivity_location()

Default sparse connectivity mode - where connectivity is initialised.

.. index:: pair: function; model_name
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afefdda5f27d55ca55d7e8e7703372785:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def model_name()

Name of the model.

.. index:: pair: function; t
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ab67386f1c84f9c828827dfa54d609e78:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def t()

Simulation time in ms.

.. index:: pair: function; timestep
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a204a3f0b579978cf3534d2c6dbb5c186:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def timestep()

Simulation time step.

.. index:: pair: function; dT
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a076181e209188da2b3592afe8a3b04f1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def dT()

Step size.

.. index:: pair: function; add_neuron_population
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a6981e562d305fa486400451c9464e775:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_neuron_population()

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

		- type of the :ref:`NeuronModels <doxid-da/dac/namespaceNeuronModels>` class as string or instance of neuron class derived from ``pygenn.genn_wrapper.NeuronModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`)

	*
		- param_space

		- dict with param values for the :ref:`NeuronModels <doxid-da/dac/namespaceNeuronModels>` class

	*
		- var_space

		- dict with initial variable values for the :ref:`NeuronModels <doxid-da/dac/namespaceNeuronModels>` class

.. index:: pair: function; add_synapse_population
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1aa0d99a65e3ca5c179feb8c120e9da252:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_synapse_population()

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

		- type of the :ref:`WeightUpdateModels <doxid-da/d80/namespaceWeightUpdateModels>` class as string or instance of weight update model class derived from ``pygenn.genn_wrapper.WeightUpdateModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`)

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

		- type of the :ref:`PostsynapticModels <doxid-db/dcb/namespacePostsynapticModels>` class as string or instance of postsynaptic model class derived from ``pygenn.genn_wrapper.PostsynapticModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138>`)

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
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ae3f4be499e6355cc917d0d3d6239815a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_current_source()

Add a current source to the GeNN model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- cs_name

		- name of the new current source

	*
		- current_source_model

		- type of the :ref:`CurrentSourceModels <doxid-d6/dd3/namespaceCurrentSourceModels>` class as string or instance of :ref:`CurrentSourceModels <doxid-d6/dd3/namespaceCurrentSourceModels>` class derived from ``pygenn.genn_wrapper.CurrentSourceModels.Custom`` (see also :ref:`pygenn.genn_model.create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`)

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
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a271d098822afa359ca392752b16683cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def build()

Finalize and build a GeNN model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- path_to_model

		- path where to place the generated model code. Defaults to the local directory.

.. index:: pair: function; load
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a8baf03355871cb66ea2368909433c22a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def load()

import the model as shared library and initialize it

.. index:: pair: function; reinitialise
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a355a09159b9a00859300e01fae12687e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def reinitialise()

reinitialise model to its original state without re-loading

.. index:: pair: function; pull_state_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a71633dae5b0ac071b255857073bbf7e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_state_from_device()

Pull state from the device for a given population.

.. index:: pair: function; pull_spikes_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a4ba17d02864196b87f79cd25f9b326e7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_spikes_from_device()

Pull spikes from the device for a given population.

.. index:: pair: function; pull_current_spikes_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1af88752774f48315be30964fb44226fb1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_current_spikes_from_device()

Pull spikes from the device for a given population.

.. index:: pair: function; pull_connectivity_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a586c5aa3caf6bc232eab54c9ea7f3218:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_connectivity_from_device()

Pull connectivity from the device for a given population.

.. index:: pair: function; pull_var_from_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ad0fa3cd33bf965b706459d21328c49a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pull_var_from_device()

Pull variable from the device for a given population.

.. index:: pair: function; push_state_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afe34bc9de940fc5ca5d23f02140caa1a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_state_to_device()

Push state to the device for a given population.

.. index:: pair: function; push_spikes_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a4109fc7d61910f2836b036e4ca80d002:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_spikes_to_device()

Push spikes to the device for a given population.

.. index:: pair: function; push_current_spikes_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1afbb2cb09e264df5615fc6e39d6fe5545:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_current_spikes_to_device()

Push current spikes to the device for a given population.

.. index:: pair: function; push_connectivity_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1ac729fc62b67c260c0d91fb9c374bab62:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_connectivity_to_device()

Push connectivity to the device for a given population.

.. index:: pair: function; push_var_to_device
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1a912ec53be14d047d9a47e3d0fd1ba938:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def push_var_to_device()

Push variable to the device for a given population.

.. index:: pair: function; end
.. _doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel_1aa1ab378068b692100af0cec3014cdf5f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def end()

Free memory.

