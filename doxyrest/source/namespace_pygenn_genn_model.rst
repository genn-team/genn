.. index:: pair: namespace; pygenn::genn_model
.. _doxid-de/d6e/namespacepygenn_1_1genn__model:

namespace pygenn::genn_model
============================

.. toctree::
	:hidden:

	class_pygenn_genn_model_GeNNModel.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace genn_model {

	// classes

	class :ref:`GeNNModel<doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>`;

	// global variables

	 :target:`backend_modules<doxid-de/d6e/namespacepygenn_1_1genn__model_1a0378993ab51a2871fc274bddf7b55bcd>`;
	 :target:`m<doxid-de/d6e/namespacepygenn_1_1genn__model_1a0ae588c769b098f97bc66450ab28011c>`;

	// global functions

	def :ref:`init_var<doxid-de/d6e/namespacepygenn_1_1genn__model_1aa2daf7a01b3f49dcbc5dba9690020bc3>`(init_var_snippet init_var_snippet, param_space param_space);

	def :ref:`init_connectivity<doxid-de/d6e/namespacepygenn_1_1genn__model_1a7fa33c9872ab0cf97d0b60fa41bf17fe>`(
		init_sparse_connect_snippet init_sparse_connect_snippet,
		param_space param_space
		);

	def :ref:`create_custom_neuron_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		derived_params derived_params = None,
		sim_code sim_code = None,
		threshold_condition_code threshold_condition_code = None,
		reset_code reset_code = None,
		support_code support_code = None,
		extra_global_params extra_global_params = None,
		additional_input_vars additional_input_vars = None,
		is_auto_refractory_required is_auto_refractory_required = None,
		custom_body custom_body = None
		);

	def :ref:`create_custom_postsynaptic_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611>`(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		derived_params derived_params = None,
		decay_code decay_code = None,
		apply_input_code apply_input_code = None,
		support_code support_code = None,
		custom_body custom_body = None
		);

	def :ref:`create_custom_weight_update_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		pre_var_name_types pre_var_name_types = None,
		post_var_name_types post_var_name_types = None,
		derived_params derived_params = None,
		sim_code sim_code = None,
		event_code event_code = None,
		learn_post_code learn_post_code = None,
		synapse_dynamics_code synapse_dynamics_code = None,
		event_threshold_condition_code event_threshold_condition_code = None,
		pre_spike_code pre_spike_code = None,
		post_spike_code post_spike_code = None,
		sim_support_code sim_support_code = None,
		learn_post_support_code learn_post_support_code = None,
		synapse_dynamics_suppport_code synapse_dynamics_suppport_code = None,
		extra_global_params extra_global_params = None,
		is_pre_spike_time_required is_pre_spike_time_required = None,
		is_post_spike_time_required is_post_spike_time_required = None,
		custom_body custom_body = None
		);

	def :ref:`create_custom_current_source_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		derived_params derived_params = None,
		injection_code injection_code = None,
		extra_global_params extra_global_params = None,
		custom_body custom_body = None
		);

	def :ref:`create_custom_model_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a603410a2e8454498c99148657de8c460>`(
		class_name class_name,
		base base,
		param_names param_names,
		var_name_types var_name_types,
		derived_params derived_params,
		custom_body custom_body
		);

	def :ref:`create_dpf_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a4877194a4aa6f04ecbc8768ec1c2de21>`(dp_func dp_func);
	def :ref:`create_cmlf_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1acd7310ba25ce86ad93d8fc54debf0e68>`(cml_func cml_func);

	def :ref:`create_custom_init_var_snippet_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff>`(
		class_name class_name,
		param_names param_names = None,
		derived_params derived_params = None,
		var_init_code var_init_code = None,
		custom_body custom_body = None
		);

	def :ref:`create_custom_sparse_connect_init_snippet_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b>`(
		class_name class_name,
		param_names param_names = None,
		derived_params derived_params = None,
		row_build_code row_build_code = None,
		row_build_state_vars row_build_state_vars = None,
		calc_max_row_len_func calc_max_row_len_func = None,
		calc_max_col_len_func calc_max_col_len_func = None,
		extra_global_params extra_global_params = None,
		custom_body custom_body = None
		);

	} // namespace genn_model
.. _details-de/d6e/namespacepygenn_1_1genn__model:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; init_var
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1aa2daf7a01b3f49dcbc5dba9690020bc3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def init_var(init_var_snippet init_var_snippet, param_space param_space)

This helper function creates a VarInit object to easily initialise a variable using a snippet.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- init_var_snippet

		- type of the :ref:`InitVarSnippet <doxid-d2/dfc/namespaceInitVarSnippet>` class as string or instance of class derived from InitVarSnippet::Custom class.

	*
		- param_space

		- dict with param values for the :ref:`InitVarSnippet <doxid-d2/dfc/namespaceInitVarSnippet>` class

.. index:: pair: function; init_connectivity
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a7fa33c9872ab0cf97d0b60fa41bf17fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def init_connectivity(
		init_sparse_connect_snippet init_sparse_connect_snippet,
		param_space param_space
		)

This helper function creates a :ref:`InitSparseConnectivitySnippet::Init <doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>` object to easily initialise connectivity using a snippet.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- init_sparse_connect_snippet

		- type of the :ref:`InitSparseConnectivitySnippet <doxid-dc/ddd/namespaceInitSparseConnectivitySnippet>` class as string or instance of class derived from InitSparseConnectivitySnippet::Custom.

	*
		- param_space

		- dict with param values for the :ref:`InitSparseConnectivitySnippet <doxid-dc/ddd/namespaceInitSparseConnectivitySnippet>` class

.. index:: pair: function; create_custom_neuron_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_neuron_class(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		derived_params derived_params = None,
		sim_code sim_code = None,
		threshold_condition_code threshold_condition_code = None,
		reset_code reset_code = None,
		support_code support_code = None,
		extra_global_params extra_global_params = None,
		additional_input_vars additional_input_vars = None,
		is_auto_refractory_required is_auto_refractory_required = None,
		custom_body custom_body = None
		)

This helper function creates a custom NeuronModel class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- class_name

		- name of the new class

	*
		- param_names

		- list of strings with param names of the model

	*
		- var_name_types

		- list of pairs of strings with varible names and types of the model

	*
		- derived_params

		- list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of a class which inherits from ``pygenn.genn_wrapper.Snippet.DerivedParamFunc``

	*
		- sim_code

		- string with the simulation code

	*
		- threshold_condition_code

		- string with the threshold condition code

	*
		- reset_code

		- string with the reset code

	*
		- support_code

		- string with the support code

	*
		- extra_global_params

		- list of pairs of strings with names and types of additional parameters

	*
		- additional_input_vars

		- list of tuples with names and types as strings and initial values of additional local input variables

	*
		- is_auto_refractory_required

		- does this model require auto-refractory logic to be generated?

	*
		- custom_body

		- dictionary with additional attributes and methods of the new class



.. rubric:: See also:

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b>`

.. index:: pair: function; create_custom_postsynaptic_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_postsynaptic_class(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		derived_params derived_params = None,
		decay_code decay_code = None,
		apply_input_code apply_input_code = None,
		support_code support_code = None,
		custom_body custom_body = None
		)

This helper function creates a custom PostsynapticModel class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- class_name

		- name of the new class

	*
		- param_names

		- list of strings with param names of the model

	*
		- var_name_types

		- list of pairs of strings with varible names and types of the model

	*
		- derived_params

		- list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of a class which inherits from ``pygenn.genn_wrapper.DerivedParamFunc``

	*
		- decay_code

		- string with the decay code

	*
		- apply_input_code

		- string with the apply input code

	*
		- support_code

		- string with the support code

	*
		- custom_body

		- dictionary with additional attributes and methods of the new class



.. rubric:: See also:

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b>`

.. index:: pair: function; create_custom_weight_update_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_weight_update_class(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		pre_var_name_types pre_var_name_types = None,
		post_var_name_types post_var_name_types = None,
		derived_params derived_params = None,
		sim_code sim_code = None,
		event_code event_code = None,
		learn_post_code learn_post_code = None,
		synapse_dynamics_code synapse_dynamics_code = None,
		event_threshold_condition_code event_threshold_condition_code = None,
		pre_spike_code pre_spike_code = None,
		post_spike_code post_spike_code = None,
		sim_support_code sim_support_code = None,
		learn_post_support_code learn_post_support_code = None,
		synapse_dynamics_suppport_code synapse_dynamics_suppport_code = None,
		extra_global_params extra_global_params = None,
		is_pre_spike_time_required is_pre_spike_time_required = None,
		is_post_spike_time_required is_post_spike_time_required = None,
		custom_body custom_body = None
		)

This helper function creates a custom WeightUpdateModel class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- class_name

		- name of the new class

	*
		- param_names

		- list of strings with param names of the model

	*
		- var_name_types

		- list of pairs of strings with variable names and types of the model

	*
		- pre_var_name_types

		- list of pairs of strings with presynaptic variable names and types of the model

	*
		- post_var_name_types

		- list of pairs of strings with postsynaptic variable names and types of the model

	*
		- derived_params

		- list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of a class which inherits from ``pygenn.genn_wrapper.DerivedParamFunc``

	*
		- sim_code

		- string with the simulation code

	*
		- event_code

		- string with the event code

	*
		- learn_post_code

		- string with the code to include in learn_synapse_post kernel/function

	*
		- synapse_dynamics_code

		- string with the synapse dynamics code

	*
		- event_threshold_condition_code

		- string with the event threshold condition code

	*
		- pre_spike_code

		- string with the code run once per spiking presynaptic neuron

	*
		- post_spike_code

		- string with the code run once per spiking postsynaptic neuron

	*
		- sim_support_code

		- string with simulation support code

	*
		- learn_post_support_code

		- string with support code for learn_synapse_post kernel/function

	*
		- synapse_dynamics_suppport_code

		- string with synapse dynamics support code

	*
		- extra_global_params

		- list of pairs of strings with names and types of additional parameters

	*
		- is_pre_spike_time_required

		- boolean, is presynaptic spike time required in any weight update kernels?

	*
		- is_post_spike_time_required

		- boolean, is postsynaptic spike time required in any weight update kernels?

	*
		- custom_body

		- dictionary with additional attributes and methods of the new class



.. rubric:: See also:

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b>`

.. index:: pair: function; create_custom_current_source_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_current_source_class(
		class_name class_name,
		param_names param_names = None,
		var_name_types var_name_types = None,
		derived_params derived_params = None,
		injection_code injection_code = None,
		extra_global_params extra_global_params = None,
		custom_body custom_body = None
		)

This helper function creates a custom NeuronModel class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- class_name

		- name of the new class

	*
		- param_names

		- list of strings with param names of the model

	*
		- var_name_types

		- list of pairs of strings with varible names and types of the model

	*
		- derived_params

		- list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of the class which inherits from ``pygenn.genn_wrapper.DerivedParamFunc``

	*
		- injection_code

		- string with the current injection code

	*
		- extra_global_params

		- list of pairs of strings with names and types of additional parameters

	*
		- custom_body

		- dictionary with additional attributes and methods of the new class



.. rubric:: See also:

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b>`

.. index:: pair: function; create_custom_model_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a603410a2e8454498c99148657de8c460:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_model_class(
		class_name class_name,
		base base,
		param_names param_names,
		var_name_types var_name_types,
		derived_params derived_params,
		custom_body custom_body
		)

This helper function completes a custom model class creation.

This part is common for all model classes and is nearly useless on its own unless you specify custom_body.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- class_name

		- name of the new class

	*
		- base

		- base class

	*
		- param_names

		- list of strings with param names of the model

	*
		- var_name_types

		- list of pairs of strings with varible names and types of the model

	*
		- derived_params

		- list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of the class which inherits from the ``pygenn.genn_wrapper.DerivedParamFunc`` class

	*
		- custom_body

		- dictionary with attributes and methods of the new class



.. rubric:: See also:

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b>`

.. index:: pair: function; create_dpf_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a4877194a4aa6f04ecbc8768ec1c2de21:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_dpf_class(dp_func dp_func)

Helper function to create derived parameter function class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- dp_func

		- a function which computes the derived parameter and takes two args "pars" (vector of double) and "dt" (double)

.. index:: pair: function; create_cmlf_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1acd7310ba25ce86ad93d8fc54debf0e68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_cmlf_class(cml_func cml_func)

Helper function to create function class for calculating sizes of matrices initialised with sparse connectivity initialisation snippet.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- cml_func

		- a function which computes the length and takes three args "num_pre" (unsigned int), "num_post" (unsigned int) and "pars" (vector of double)

.. index:: pair: function; create_custom_init_var_snippet_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_init_var_snippet_class(
		class_name class_name,
		param_names param_names = None,
		derived_params derived_params = None,
		var_init_code var_init_code = None,
		custom_body custom_body = None
		)

This helper function creates a custom :ref:`InitVarSnippet <doxid-d2/dfc/namespaceInitVarSnippet>` class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- class_name

		- name of the new class

	*
		- param_names

		- list of strings with param names of the model

	*
		- derived_params

		- list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of the ``pygenn.genn_wrapper.DerivedParamFunc`` ` class

	*
		- var_init_code

		- string with the variable initialization code

	*
		- custom_body

		- dictionary with additional attributes and methods of the new class



.. rubric:: See also:

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b>`

.. index:: pair: function; create_custom_sparse_connect_init_snippet_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1acd4074f475e3e48c21d1c31d1a28597b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_sparse_connect_init_snippet_class(
		class_name class_name,
		param_names param_names = None,
		derived_params derived_params = None,
		row_build_code row_build_code = None,
		row_build_state_vars row_build_state_vars = None,
		calc_max_row_len_func calc_max_row_len_func = None,
		calc_max_col_len_func calc_max_col_len_func = None,
		extra_global_params extra_global_params = None,
		custom_body custom_body = None
		)

This helper function creates a custom :ref:`InitSparseConnectivitySnippet <doxid-dc/ddd/namespaceInitSparseConnectivitySnippet>` class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- class_name

		- name of the new class

	*
		- param_names

		- list of strings with param names of the model

	*
		- derived_params

		- list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of the class which inherits from ``pygenn.genn_wrapper.DerivedParamFunc``

	*
		- row_build_code

		- string with row building initialization code

	*
		- row_build_state_vars

		- list of tuples of state variables, their types and their initial values to use across row building loop

	*
		- calc_max_row_len_func

		- instance of class inheriting from CalcMaxLengthFunc used to calculate maximum row length of synaptic matrix

	*
		- calc_max_col_len_func

		- instance of class inheriting from CalcMaxLengthFunc used to calculate maximum col length of synaptic matrix

	*
		- extra_global_params

		- list of pairs of strings with names and types of additional parameters

	*
		- custom_body

		- dictionary with additional attributes and methods of the new class



.. rubric:: See also:

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a80f3b56cb2dc934ea04ed15a629c7db9>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af8c5efd0096b17d61e13463b1cc73878>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a091ea4fe261201327c877528287f9611>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a1cd795d295c4cc4f05968f04dbf5b9d3>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a432e3d1d2c6c4745a1d79ee147b88fff>`

