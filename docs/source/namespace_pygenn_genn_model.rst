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

	tuple :target:`backend_modules<doxid-de/d6e/namespacepygenn_1_1genn__model_1a82394dbc72a8c94cdf12968807e4307d>`;
	tuple :target:`m<doxid-de/d6e/namespacepygenn_1_1genn__model_1a22227864f213a2a35fec9c99bc490a16>`;

	// global functions

	def :ref:`init_var<doxid-de/d6e/namespacepygenn_1_1genn__model_1a07f8ce7769b538aed7e8f81762f025b5>`();
	def :ref:`init_connectivity<doxid-de/d6e/namespacepygenn_1_1genn__model_1aa7165e8d9a7156dc8685fff376d51270>`();
	def :ref:`create_custom_neuron_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`();
	def :ref:`create_custom_postsynaptic_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138>`();
	def :ref:`create_custom_weight_update_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`();
	def :ref:`create_custom_current_source_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`();
	def :ref:`create_custom_model_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1ab53f2b73940e694cfb903f66eb36b268>`();
	def :ref:`create_dpf_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1ac168d515f72c3f48eddabbd78d34166b>`();
	def :ref:`create_cmlf_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1ac5cff0f737f177c327eaf3e3e9603177>`();
	def :ref:`create_custom_init_var_snippet_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0>`();
	def :ref:`create_custom_sparse_connect_init_snippet_class<doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0>`();

	} // namespace genn_model
.. _details-de/d6e/namespacepygenn_1_1genn__model:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; init_var
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a07f8ce7769b538aed7e8f81762f025b5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def init_var()

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
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1aa7165e8d9a7156dc8685fff376d51270:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def init_connectivity()

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
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_neuron_class()

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

		- 
		  list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of a class which inherits from
		  
		  .. ref-code-block:: cpp
		  
		  	                                   ``pygenn.genn_wrapper.Snippet.DerivedParamFunc``
		  	@param     sim_code    string with the simulation code
		  	@param     threshold_condition_code    string with the threshold condition code
		  	@param     reset_code  string with the reset code
		  	@param     support_code    string with the support code
		  	@param     extra_global_params list of pairs of strings with names and
		  
		  types of additional parameters

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

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0>`

.. index:: pair: function; create_custom_postsynaptic_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_postsynaptic_class()

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

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0>`

.. index:: pair: function; create_custom_weight_update_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_weight_update_class()

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

		- 
		  list of pairs, where the first member is string with name of the derived parameter and the second MUST be an instance of a class which inherits from
		  
		  .. ref-code-block:: cpp
		  
		  	                                       ``pygenn.genn_wrapper.DerivedParamFunc``
		  	@param     sim_code    string with the simulation code
		  	@param     event_code  string with the event code
		  	@param     learn_post_code string with the code to include in
		  
		  learn_synapse_post kernel/function

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

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0>`

.. index:: pair: function; create_custom_current_source_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_current_source_class()

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

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0>`

.. index:: pair: function; create_custom_model_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1ab53f2b73940e694cfb903f66eb36b268:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_model_class()

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

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0>`

.. index:: pair: function; create_dpf_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1ac168d515f72c3f48eddabbd78d34166b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_dpf_class()

Helper function to create derived parameter function class.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- dp_func

		- a function which computes the derived parameter and takes two args "pars" (vector of double) and "dt" (double)

.. index:: pair: function; create_cmlf_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1ac5cff0f737f177c327eaf3e3e9603177:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_cmlf_class()

Helper function to create function class for calculating sizes of matrices initialised with sparse connectivity initialisation snippet.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- cml_func

		- a function which computes the length and takes three args "num_pre" (unsigned int), "num_post" (unsigned int) and "pars" (vector of double)

.. index:: pair: function; create_custom_init_var_snippet_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_init_var_snippet_class()

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

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`

:ref:`create_custom_sparse_connect_init_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0>`

.. index:: pair: function; create_custom_sparse_connect_init_snippet_class
.. _doxid-de/d6e/namespacepygenn_1_1genn__model_1a85fee2c4f7423b65ac44af05d1c721c0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def create_custom_sparse_connect_init_snippet_class()

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

:ref:`create_custom_neuron_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a853b9227af2ed40a4b81b7c891452b>`

:ref:`create_custom_weight_update_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a9a28377fbeef1d2e3b4a5ddf8f763af3>`

:ref:`create_custom_postsynaptic_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a67332bdc9d851f2de537bb1b8ee81138>`

:ref:`create_custom_current_source_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1a940817f86b8a6e16139ec7beaf2e2a9a>`

:ref:`create_custom_init_var_snippet_class <doxid-de/d6e/namespacepygenn_1_1genn__model_1af842768d9ca6333c64063900e479e1b0>`

