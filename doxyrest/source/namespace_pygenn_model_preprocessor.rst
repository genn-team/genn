.. index:: pair: namespace; pygenn::model_preprocessor
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor:

namespace pygenn::model_preprocessor
====================================

.. toctree::
	:hidden:

	class_pygenn_model_preprocessor_ExtraGlobalVariable.rst
	class_pygenn_model_preprocessor_Variable.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace model_preprocessor {

	// classes

	class :ref:`ExtraGlobalVariable<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable>`;
	class :ref:`Variable<doxid-d5/de5/classpygenn_1_1model__preprocessor_1_1Variable>`;

	// global variables

	dictionary :target:`genn_to_numpy_types<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a61d7be69c20babd2663997b0f5769e10>`;

	// global functions

	def :ref:`prepare_model<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1afa2002972d2398cc6727c36712c27a25>`();
	def :ref:`prepare_snippet<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1af8787ed2b94336e2f15d1bb2a7525ca2>`();
	def :ref:`is_model_valid<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aed65dd8c7c532f1d68afc6792082f8f8>`();
	def :ref:`param_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aa5dcb3499ba649215fc1de5a5760347c>`();
	def :ref:`param_space_to_val_vec<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aace69bd4664d6b8420bcfa717b999ef1>`();
	def :ref:`var_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a3b22825cc9feaa2db636d339f4af547f>`();
	def :ref:`pre_var_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aa92c968e8193bf3b61aab69eccc9fc39>`();
	def :ref:`post_var_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a474973e600f1cd24e3156cba8b8cf355>`();

	} // namespace model_preprocessor
.. _details-d0/d17/namespacepygenn_1_1model__preprocessor:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; prepare_model
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1afa2002972d2398cc6727c36712c27a25:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def prepare_model()

Prepare a model by checking its validity and extracting information about variables and parameters.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- 
		  string or instance of a class derived from ``pygenn.genn_wrapper.NeuronModels.Custom`` or ``pygenn.genn_wrapper.WeightUpdateModels.Custom`` or
		  
		  .. ref-code-block:: cpp
		  
		  	                       ``pygenn.genn_wrapper.CurrentSourceModels.Custom``
		  	@param     param_space dict with model parameters
		  	@param     var_space   dict with model variables
		  	@param     pre_var_space   optional dict with (weight update) model
		  
		  presynaptic variables

	*
		- post_var_space

		- optional dict with (weight update) model postsynaptic variables

	*
		- model_family

		- 
		  ``pygenn.genn_wrapper.NeuronModels`` or ``pygenn.genn_wrapper.WeightUpdateModels`` or
		  
		  .. ref-code-block:: cpp
		  
		  	                       ``pygenn.genn_wrapper.CurrentSourceModels``
		  	
		  	@return
		  
		  tuple consisting of (model instance, model type, model parameter names, model parameters, list of variable names, dict mapping names of variables to instances of class :ref:`Variable <doxid-d5/de5/classpygenn_1_1model__preprocessor_1_1Variable>`)

.. index:: pair: function; prepare_snippet
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1af8787ed2b94336e2f15d1bb2a7525ca2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def prepare_snippet()

Prepare a snippet by checking its validity and extracting information about parameters.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- snippet

		- 
		  string or instance of a class derived from ``pygenn.genn_wrapper.InitVarSnippet.Custom`` or
		  
		  .. ref-code-block:: cpp
		  
		  	                       ``pygenn.genn_wrapper.InitSparseConnectivitySnippet.Custom``
		  	@param     param_space dict with model parameters
		  	@param     snippet_family  ``pygenn.genn_wrapper.InitVarSnippet`` or
		  	                       ``pygenn.genn_wrapper.InitSparseConnectivitySnippet``
		  	
		  	@return
		  
		  tuple consisting of (snippet instance, snippet type, snippet parameter names, snippet parameters)

.. index:: pair: function; is_model_valid
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aed65dd8c7c532f1d68afc6792082f8f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_model_valid()

Check whether the model is valid, i.e is native or derived from model_family.Custom.

Raises ValueError if model is not valid (i.e. is not custom and is not natively available)



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- string or instance of model_family.Custom

	*
		- model_family

		- model family (:ref:`NeuronModels <doxid-da/dac/namespaceNeuronModels>`, :ref:`WeightUpdateModels <doxid-da/d80/namespaceWeightUpdateModels>` or :ref:`PostsynapticModels <doxid-db/dcb/namespacePostsynapticModels>`) to which model should belong to



.. rubric:: Returns:

instance of the model and its type as string

.. index:: pair: function; param_space_to_vals
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aa5dcb3499ba649215fc1de5a5760347c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def param_space_to_vals()

Convert a param_space dict to ParamValues.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- instance of the model

	*
		- param_space

		- dict with parameters



.. rubric:: Returns:

native model's ParamValues

.. index:: pair: function; param_space_to_val_vec
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aace69bd4664d6b8420bcfa717b999ef1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def param_space_to_val_vec()

Convert a param_space dict to a std::vector<double>



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- instance of the model

	*
		- param_space

		- dict with parameters



.. rubric:: Returns:

native vector of parameters

.. index:: pair: function; var_space_to_vals
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a3b22825cc9feaa2db636d339f4af547f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def var_space_to_vals()

Convert a var_space dict to VarValues.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- instance of the model

	*
		- var_space

		- dict with Variables



.. rubric:: Returns:

native model's VarValues

.. index:: pair: function; pre_var_space_to_vals
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aa92c968e8193bf3b61aab69eccc9fc39:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pre_var_space_to_vals()

Convert a var_space dict to PreVarValues.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- instance of the weight update model

	*
		- var_space

		- dict with Variables



.. rubric:: Returns:

native model's VarValues

.. index:: pair: function; post_var_space_to_vals
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a474973e600f1cd24e3156cba8b8cf355:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def post_var_space_to_vals()

Convert a var_space dict to PostVarValues.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- instance of the weight update model

	*
		- var_space

		- dict with Variables



.. rubric:: Returns:

native model's VarValues

