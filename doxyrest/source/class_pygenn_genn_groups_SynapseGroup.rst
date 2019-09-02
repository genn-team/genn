.. index:: pair: class; pygenn::genn_groups::SynapseGroup
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup:

class pygenn::genn_groups::SynapseGroup
=======================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Class representing synaptic connection between two groups of neurons. :ref:`More...<details-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	class SynapseGroup: public :ref:`pygenn::genn_groups::Group<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group>`
	{
	public:
		// fields
	
		 :target:`connections_set<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1af92bed3adac69c3fe0701aa312fd07b5>`;
		 :target:`w_update<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a7cb6fa144610d947d3226a7baa610043>`;
		 :target:`postsyn<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a53f2a6f22edff8cd21b8b66edfe94432>`;
		 :target:`src<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a146bc27cb532ff83bdfc3b8ee77a4837>`;
		 :target:`trg<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a13fe205aafc106f284fbc43ea2cb5399>`;
		 :target:`psm_vars<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a1e4d53dd8d45f5af69098c3def912bbd>`;
		 :target:`pre_vars<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1acf0622c21f3ab6a873acdd8a0d513874>`;
		 :target:`post_vars<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a767c5dac123b65eed9ebeded458db58f>`;
		 :target:`connectivity_initialiser<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a48893828585d570d55f9c488e9690dd2>`;
		 :target:`synapse_order<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a36817bb142198afef42d4b76a45651ab>`;
		 :target:`ind<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ad9b056b4a4620bb9f186120e9e8e40cb>`;
		 :target:`row_lengths<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1adeae076892f9c8028bb381f4691ce6c3>`;
		 :target:`pop<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a29fb9f595bd4e6e4289139f6c9cfb502>`;

		// methods
	
		def :ref:`__init__<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ada93239064c29c7587a3cc23c7a26b1b>`(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name);
		def :ref:`num_synapses<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a47259ad1e1fb0b61ec07652dd272cc8e>`(self self);
		def :ref:`weight_update_var_size<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1abc7df7f3c37f091bea75d067d1e87835>`(self self);
		def :target:`max_row_length<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a2208fcc9522823d65c1f95775fa27982>`(self self);
		def :ref:`set_psm_var<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a419fc2b646104bd7e43b28a5be8fc098>`(self self, var_name var_name, values values);
		def :ref:`set_pre_var<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8f0673bd4445ab35ae37af0abb805afd>`(self self, var_name var_name, values values);
		def :ref:`set_post_var<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a3c07032d789c999574d5832999145a5c>`(self self, var_name var_name, values values);
	
		def :ref:`set_weight_update<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ab8253488d655ab2e1f6291689a973ee1>`(
			self self,
			model model,
			param_space param_space,
			var_space var_space,
			pre_var_space pre_var_space,
			post_var_space post_var_space
			);
	
		def :ref:`set_post_syn<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a6e0b3abf2a3024d428804c028a950cb8>`(self self, model model, param_space param_space, var_space var_space);
	
		def :target:`get_var_values<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1af4d5e6078a2f7e231a869909731d90ab>`(
			self self,
			var_name var_name
			);
	
		def :target:`is_connectivity_init_required<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a50c4e431ff060ee5156ee78113c75908>`(self self);
		def :ref:`matrix_type<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8a83848fea29eb821b094759b7448891>`(self self);
	
		def :target:`matrix_type<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a2f68c883612f6cba26d9ce34a2572c2f>`(
			self self,
			matrix_type matrix_type
			);
	
		def :ref:`is_ragged<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aeda75ea52b1cc1c75444a0371fe4b664>`(self self);
		def :ref:`is_bitmask<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8bca5a6e84b636d485391f5a17555fbb>`(self self);
		def :ref:`is_dense<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ac4de8bf08ebcf36115712b639dee0b3f>`(self self);
		def :ref:`has_individual_synapse_vars<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a0433608d422386e67a34c0d846097844>`(self self);
		def :ref:`has_individual_postsynaptic_vars<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a4fe45637de528f76c288adb09a4c3b03>`(self self);
		def :ref:`set_sparse_connections<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a89c87f35af246051db899da4c6b5caf3>`(self self, pre_indices pre_indices, post_indices post_indices);
		def :ref:`get_sparse_pre_inds<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ad64df56c8184a34c85f256e09be9063f>`(self self);
		def :ref:`get_sparse_post_inds<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8e3a0f310ee5ce1afcfffea3eb0fe67c>`(self self);
		def :ref:`set_connected_populations<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a6d22f60a1997a803c8fe4de2ee59d926>`(self self, source source, target target);
		def :ref:`add_to<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a7661170e06a9fbc945f697599618a285>`(self self, model_spec model_spec, delay_steps delay_steps);
		def :ref:`add_extra_global_param<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ac5806934aaf2a806aa7d5e6199f07335>`(self self, param_name param_name, param_values param_values);
	
		def :target:`load<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a1b67ba944d1622a571a79a1040918419>`(
			self self,
			slm slm,
			scalar scalar
			);
	
		def :ref:`reinitialise<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aec3de8b797bcc7ef6ec7d2c7f9a41e6f>`(self self, slm slm, scalar scalar);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// fields
	
		 :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>`;
		 :ref:`vars<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1ad8e33584773714170465d5166c9c5e3e>`;
		 :ref:`extra_global_params<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a646ab45304e93c1cec854df96e8fb197>`;

		// methods
	
		def :ref:`__init__<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a44c2cfa0fba7209cc89c0e70b3882f71>`(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name);
		def :ref:`set_var<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a5b79e0c32a3f3c3e03c2a03baa6e13d3>`(self self, var_name var_name, values values);

.. _details-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Class representing synaptic connection between two groups of neurons.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ada93239064c29c7587a3cc23c7a26b1b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name)

Init :ref:`SynapseGroup <doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- name

		- string name of the group

.. index:: pair: function; num_synapses
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a47259ad1e1fb0b61ec07652dd272cc8e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def num_synapses(self self)

Number of synapses in group.

.. index:: pair: function; weight_update_var_size
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1abc7df7f3c37f091bea75d067d1e87835:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def weight_update_var_size(self self)

Size of each weight update variable.

.. index:: pair: function; set_psm_var
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a419fc2b646104bd7e43b28a5be8fc098:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_psm_var(self self, var_name var_name, values values)

Set values for a postsynaptic model variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- var_name

		- string with the name of the postsynaptic model variable

	*
		- values

		- iterable or a single value

.. index:: pair: function; set_pre_var
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8f0673bd4445ab35ae37af0abb805afd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_pre_var(self self, var_name var_name, values values)

Set values for a presynaptic variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- var_name

		- string with the name of the presynaptic variable

	*
		- values

		- iterable or a single value

.. index:: pair: function; set_post_var
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a3c07032d789c999574d5832999145a5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_post_var(self self, var_name var_name, values values)

Set values for a postsynaptic variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- var_name

		- string with the name of the presynaptic variable

	*
		- values

		- iterable or a single value

.. index:: pair: function; set_weight_update
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ab8253488d655ab2e1f6291689a973ee1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_weight_update(
		self self,
		model model,
		param_space param_space,
		var_space var_space,
		pre_var_space pre_var_space,
		post_var_space post_var_space
		)

Set weight update model, its parameters and initial variables.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- type as string of intance of the model

	*
		- param_space

		- dict with model parameters

	*
		- var_space

		- dict with model variables

	*
		- pre_var_space

		- dict with model presynaptic variables

	*
		- post_var_space

		- dict with model postsynaptic variables

.. index:: pair: function; set_post_syn
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a6e0b3abf2a3024d428804c028a950cb8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_post_syn(
		self self,
		model model,
		param_space param_space,
		var_space var_space
		)

Set postsynaptic model, its parameters and initial variables.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- type as string of intance of the model

	*
		- param_space

		- dict with model parameters

	*
		- var_space

		- dict with model variables

.. index:: pair: function; matrix_type
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8a83848fea29eb821b094759b7448891:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def matrix_type(self self)

Type of the projection matrix.

.. index:: pair: function; is_ragged
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aeda75ea52b1cc1c75444a0371fe4b664:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_ragged(self self)

Tests whether synaptic connectivity uses Ragged format.

.. index:: pair: function; is_bitmask
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8bca5a6e84b636d485391f5a17555fbb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_bitmask(self self)

Tests whether synaptic connectivity uses Bitmask format.

.. index:: pair: function; is_dense
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ac4de8bf08ebcf36115712b639dee0b3f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_dense(self self)

Tests whether synaptic connectivity uses dense format.

.. index:: pair: function; has_individual_synapse_vars
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a0433608d422386e67a34c0d846097844:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def has_individual_synapse_vars(self self)

Tests whether synaptic connectivity has individual weights.

.. index:: pair: function; has_individual_postsynaptic_vars
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a4fe45637de528f76c288adb09a4c3b03:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def has_individual_postsynaptic_vars(self self)

Tests whether synaptic connectivity has individual postsynaptic model variables.

.. index:: pair: function; set_sparse_connections
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a89c87f35af246051db899da4c6b5caf3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_sparse_connections(
		self self,
		pre_indices pre_indices,
		post_indices post_indices
		)

Set ragged format connections between two groups of neurons.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pre_indices

		- ndarray of presynaptic indices

	*
		- post_indices

		- ndarray of postsynaptic indices

.. index:: pair: function; get_sparse_pre_inds
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ad64df56c8184a34c85f256e09be9063f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def get_sparse_pre_inds(self self)

Get presynaptic indices of synapse group connections.



.. rubric:: Returns:

ndarray of presynaptic indices

.. index:: pair: function; get_sparse_post_inds
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8e3a0f310ee5ce1afcfffea3eb0fe67c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def get_sparse_post_inds(self self)

Get postsynaptic indices of synapse group connections.



.. rubric:: Returns:

ndarrays of postsynaptic indices

.. index:: pair: function; set_connected_populations
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a6d22f60a1997a803c8fe4de2ee59d926:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_connected_populations(self self, source source, target target)

Set two groups of neurons connected by this :ref:`SynapseGroup <doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- source

		- string name of the presynaptic neuron group

	*
		- target

		- string name of the postsynaptic neuron group

.. index:: pair: function; add_to
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a7661170e06a9fbc945f697599618a285:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_to(self self, model_spec model_spec, delay_steps delay_steps)

Add this :ref:`SynapseGroup <doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>` to the a model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model_spec

		- ``:ref:`pygenn.genn_model.GeNNModel <doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>``` to add to

	*
		- delay_steps

		- number of axonal delay timesteps to simulate for this synapse group

.. index:: pair: function; add_extra_global_param
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ac5806934aaf2a806aa7d5e6199f07335:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_extra_global_param(
		self self,
		param_name param_name,
		param_values param_values
		)

Add extra global parameter.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- param_name

		- string with the name of the extra global parameter

	*
		- param_values

		- iterable or a single value

.. index:: pair: function; reinitialise
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aec3de8b797bcc7ef6ec7d2c7f9a41e6f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def reinitialise(self self, slm slm, scalar scalar)

Reinitialise synapse group.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- slm

		- SharedLibraryModel instance for acccessing variables

	*
		- scalar

		- String specifying "scalar" type

