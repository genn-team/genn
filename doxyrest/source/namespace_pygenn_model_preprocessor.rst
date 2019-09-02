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

	def :ref:`prepare_model<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a029f48e8c8c01ba4a24e11ff4d578d11>`(
		model model,
		param_space param_space,
		var_space var_space,
		pre_var_space pre_var_space = None,
		post_var_space post_var_space = None,
		model_family model_family = None
		);

	def :ref:`prepare_snippet<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aef83afc60aa51f177264dcb416a6035c>`(snippet snippet, param_space param_space, snippet_family snippet_family);
	def :ref:`is_model_valid<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a8394727291320f68735c9ad636362853>`(model model, model_family model_family);
	def :ref:`param_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1ab095322336d228b80db47a280f435ba2>`(model model, param_space param_space);
	def :ref:`param_space_to_val_vec<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a93d693245c53a02c06c201f5769500b5>`(model model, param_space param_space);
	def :ref:`var_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a4ece878baa9619ecfcbdf5770e5a4eec>`(model model, var_space var_space);
	def :ref:`pre_var_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aa3a43765e294bbd7fe4976a3b377b733>`(model model, var_space var_space);
	def :ref:`post_var_space_to_vals<doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a5d924093505fd70d9261b35a0f9c6a6d>`(model model, var_space var_space);

	} // namespace model_preprocessor
.. _details-d0/d17/namespacepygenn_1_1model__preprocessor:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; prepare_model
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a029f48e8c8c01ba4a24e11ff4d578d11:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def prepare_model(
		model model,
		param_space param_space,
		var_space var_space,
		pre_var_space pre_var_space = None,
		post_var_space post_var_space = None,
		model_family model_family = None
		)

Prepare a model by checking its validity and extracting information about variables and parameters.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- string or instance of a class derived from ``pygenn.genn_wrapper.NeuronModels.Custom`` or ``pygenn.genn_wrapper.WeightUpdateModels.Custom`` or ``pygenn.genn_wrapper.CurrentSourceModels.Custom``

	*
		- param_space

		- dict with model parameters

	*
		- var_space

		- dict with model variables

	*
		- pre_var_space

		- optional dict with (weight update) model presynaptic variables

	*
		- post_var_space

		- optional dict with (weight update) model postsynaptic variables

	*
		- model_family

		- ``pygenn.genn_wrapper.NeuronModels`` or ``pygenn.genn_wrapper.WeightUpdateModels`` or ``pygenn.genn_wrapper.CurrentSourceModels``



.. rubric:: Returns:

tuple consisting of (model instance, model type, model parameter names, model parameters, list of variable names, dict mapping names of variables to instances of class :ref:`Variable <doxid-d5/de5/classpygenn_1_1model__preprocessor_1_1Variable>`)

.. index:: pair: function; prepare_snippet
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aef83afc60aa51f177264dcb416a6035c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def prepare_snippet(
		snippet snippet,
		param_space param_space,
		snippet_family snippet_family
		)

Prepare a snippet by checking its validity and extracting information about parameters.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- snippet

		- string or instance of a class derived from ``pygenn.genn_wrapper.InitVarSnippet.Custom`` or ``pygenn.genn_wrapper.InitSparseConnectivitySnippet.Custom``

	*
		- param_space

		- dict with model parameters

	*
		- snippet_family

		- ``pygenn.genn_wrapper.InitVarSnippet`` or ``pygenn.genn_wrapper.InitSparseConnectivitySnippet``



.. rubric:: Returns:

tuple consisting of (snippet instance, snippet type, snippet parameter names, snippet parameters)

.. index:: pair: function; is_model_valid
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a8394727291320f68735c9ad636362853:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def is_model_valid(model model, model_family model_family)

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
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1ab095322336d228b80db47a280f435ba2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def param_space_to_vals(model model, param_space param_space)

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
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a93d693245c53a02c06c201f5769500b5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def param_space_to_val_vec(model model, param_space param_space)

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
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a4ece878baa9619ecfcbdf5770e5a4eec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def var_space_to_vals(model model, var_space var_space)

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
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1aa3a43765e294bbd7fe4976a3b377b733:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def pre_var_space_to_vals(model model, var_space var_space)

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
.. _doxid-d0/d17/namespacepygenn_1_1model__preprocessor_1a5d924093505fd70d9261b35a0f9c6a6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def post_var_space_to_vals(model model, var_space var_space)

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

