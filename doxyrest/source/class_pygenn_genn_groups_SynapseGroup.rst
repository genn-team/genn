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
	
		def :ref:`__init__<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a0d032c2d21481081009c6459a2ddea58>`();
		def :ref:`num_synapses<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aa2ddf1e6081e7c01dcf90f7e49b0fc57>`();
		def :ref:`weight_update_var_size<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a974d1f6d6b629717437fab8315054394>`();
		def :target:`max_row_length<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aadaaf0db626d902a5fb4754a3cf25429>`();
		def :ref:`set_psm_var<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1af1dffa81fe459f58a21d13e0b0fee090>`();
		def :ref:`set_pre_var<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a7a76cd45fd93d32ceede3f924e5b9b78>`();
		def :ref:`set_post_var<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ae5795835b89ea6364ffee9a30465a94a>`();
		def :ref:`set_weight_update<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a316c90d7173183ddbb9245b202449c73>`();
		def :ref:`set_post_syn<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a867e720af53ba523d909955b3dd48e95>`();
		def :target:`get_var_values<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ae81d8f37d59bf673c07dd795b0334291>`();
		def :target:`is_connectivity_init_required<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ad8d6080f554065f0e74a84db4cd91182>`();
		def :ref:`matrix_type<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1abec88bd8fdaa7f11cf9acb8b74c8b144>`();
		def :ref:`is_ragged<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ad403c3cdc54e346e39ddceb41fba37b2>`();
		def :ref:`is_bitmask<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a50b6bc9aa3d1a98d99ee749ab9b6ce52>`();
		def :ref:`is_dense<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a84882a96109d7cfade7a8c0b84d61024>`();
		def :ref:`has_individual_synapse_vars<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aca255452742fe41b1ab633b4b9b66445>`();
		def :ref:`has_individual_postsynaptic_vars<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a37f0965d3a9341d1bc7ce64e291fa005>`();
		def :ref:`set_sparse_connections<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8f66d8bc1734bd96051c0eeb9000704d>`();
		def :ref:`get_sparse_pre_inds<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a96ffea99169c3a6f4c489f956571b1d1>`();
		def :ref:`get_sparse_post_inds<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a943406d5e90d44bb8cb3c1c4686b2fe5>`();
		def :ref:`set_connected_populations<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1af761a714129ed4b9d0b3fd103746dadd>`();
		def :ref:`add_to<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a4b017a4e0dbc0820f668ecf4110deae7>`();
		def :ref:`add_extra_global_param<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a32610289a4697b29ac38e19ac4010018>`();
		def :target:`load<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ac980aa0e13d5d4db638d36a5496588d8>`();
		def :ref:`reinitialise<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a59cd1b532d1009f204aae96d676088ca>`();
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
	
		def :ref:`__init__<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1ac4f6b8f8fb67862785fd508f23d50140>`();
		def :ref:`set_var<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a9a44ddf9d465e2f272d8e5d16aa82c54>`();

.. _details-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Class representing synaptic connection between two groups of neurons.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a0d032c2d21481081009c6459a2ddea58:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__()

Init :ref:`SynapseGroup <doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- name

		- string name of the group

.. index:: pair: function; num_synapses
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aa2ddf1e6081e7c01dcf90f7e49b0fc57:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def num_synapses()

Number of synapses in group.

.. index:: pair: function; weight_update_var_size
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a974d1f6d6b629717437fab8315054394:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def weight_update_var_size()

Size of each weight update variable.

.. index:: pair: function; set_psm_var
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1af1dffa81fe459f58a21d13e0b0fee090:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_psm_var()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a7a76cd45fd93d32ceede3f924e5b9b78:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_pre_var()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ae5795835b89ea6364ffee9a30465a94a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_post_var()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a316c90d7173183ddbb9245b202449c73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_weight_update()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a867e720af53ba523d909955b3dd48e95:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_post_syn()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1abec88bd8fdaa7f11cf9acb8b74c8b144:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def matrix_type()

Type of the projection matrix.

.. index:: pair: function; is_ragged
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1ad403c3cdc54e346e39ddceb41fba37b2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_ragged()

Tests whether synaptic connectivity uses Ragged format.

.. index:: pair: function; is_bitmask
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a50b6bc9aa3d1a98d99ee749ab9b6ce52:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_bitmask()

Tests whether synaptic connectivity uses Bitmask format.

.. index:: pair: function; is_dense
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a84882a96109d7cfade7a8c0b84d61024:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_dense()

Tests whether synaptic connectivity uses dense format.

.. index:: pair: function; has_individual_synapse_vars
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1aca255452742fe41b1ab633b4b9b66445:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def has_individual_synapse_vars()

Tests whether synaptic connectivity has individual weights.

.. index:: pair: function; has_individual_postsynaptic_vars
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a37f0965d3a9341d1bc7ce64e291fa005:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def has_individual_postsynaptic_vars()

Tests whether synaptic connectivity has individual postsynaptic model variables.

.. index:: pair: function; set_sparse_connections
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a8f66d8bc1734bd96051c0eeb9000704d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_sparse_connections()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a96ffea99169c3a6f4c489f956571b1d1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def get_sparse_pre_inds()

Get presynaptic indices of synapse group connections.



.. rubric:: Returns:

ndarray of presynaptic indices

.. index:: pair: function; get_sparse_post_inds
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a943406d5e90d44bb8cb3c1c4686b2fe5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def get_sparse_post_inds()

Get postsynaptic indices of synapse group connections.



.. rubric:: Returns:

ndarrays of postsynaptic indices

.. index:: pair: function; set_connected_populations
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1af761a714129ed4b9d0b3fd103746dadd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_connected_populations()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a4b017a4e0dbc0820f668ecf4110deae7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_to()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a32610289a4697b29ac38e19ac4010018:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_extra_global_param()

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
.. _doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup_1a59cd1b532d1009f204aae96d676088ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def reinitialise()

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

