.. index:: pair: namespace; CodeGenerator
.. _doxid-d0/d02/namespaceCodeGenerator:

namespace CodeGenerator
=======================

.. toctree::
	:hidden:

	namespace_CodeGenerator_CUDA.rst
	namespace_CodeGenerator_SingleThreadedCPU.rst
	struct_CodeGenerator_FunctionTemplate.rst
	struct_CodeGenerator_NameIterCtx.rst
	struct_CodeGenerator_PreferencesBase.rst
	class_CodeGenerator_BackendBase.rst
	class_CodeGenerator_CodeStream.rst
	class_CodeGenerator_MemAlloc.rst
	class_CodeGenerator_StructNameConstIter.rst
	class_CodeGenerator_Substitutions.rst
	class_CodeGenerator_TeeBuf.rst
	class_CodeGenerator_TeeStream.rst

Overview
~~~~~~~~

Helper class for generating code - automatically inserts brackets, indents etc. :ref:`More...<details-d0/d02/namespaceCodeGenerator>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace CodeGenerator {

	// namespaces

	namespace :ref:`CodeGenerator::CUDA<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA>`;
		namespace :ref:`CodeGenerator::CUDA::Optimiser<doxid-d9/d85/namespaceCodeGenerator_1_1CUDA_1_1Optimiser>`;
		namespace :ref:`CodeGenerator::CUDA::PresynapticUpdateStrategy<doxid-da/d97/namespaceCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy>`;
		namespace :ref:`CodeGenerator::CUDA::Utils<doxid-d0/dd2/namespaceCodeGenerator_1_1CUDA_1_1Utils>`;
	namespace :ref:`CodeGenerator::SingleThreadedCPU<doxid-db/d8c/namespaceCodeGenerator_1_1SingleThreadedCPU>`;
		namespace :ref:`CodeGenerator::SingleThreadedCPU::Optimiser<doxid-d0/de6/namespaceCodeGenerator_1_1SingleThreadedCPU_1_1Optimiser>`;

	// typedefs

	typedef :ref:`NameIterCtx<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx>`<:ref:`Models::Base::VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>`> :target:`VarNameIterCtx<doxid-d0/d02/namespaceCodeGenerator_1a327150e79edc83fcc3645a7e93a38e0b>`;
	typedef :ref:`NameIterCtx<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx>`<:ref:`Snippet::Base::EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>`> :target:`EGPNameIterCtx<doxid-d0/d02/namespaceCodeGenerator_1a06cc2a3ea03d94368cab4cb706677fcf>`;
	typedef :ref:`NameIterCtx<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx>`<:ref:`Snippet::Base::DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>`> :target:`DerivedParamNameIterCtx<doxid-d0/d02/namespaceCodeGenerator_1a2f5a042f1f8291773cbfbead4f5fa054>`;
	typedef :ref:`NameIterCtx<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx>`<:ref:`Snippet::Base::ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>`> :target:`ParamValIterCtx<doxid-d0/d02/namespaceCodeGenerator_1a9ef00e0eb2bc78a53bcfa264a1f417c5>`;

	// structs

	struct :ref:`FunctionTemplate<doxid-dc/df1/structCodeGenerator_1_1FunctionTemplate>`;

	template <typename Container>
	struct :ref:`NameIterCtx<doxid-df/d6f/structCodeGenerator_1_1NameIterCtx>`;

	struct :ref:`PreferencesBase<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase>`;

	// classes

	class :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`;
	class :ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`;
	class :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>`;

	template <typename BaseIter>
	class :ref:`StructNameConstIter<doxid-d7/d76/classCodeGenerator_1_1StructNameConstIter>`;

	class :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`;
	class :ref:`TeeBuf<doxid-d8/d5e/classCodeGenerator_1_1TeeBuf>`;
	class :ref:`TeeStream<doxid-d7/d71/classCodeGenerator_1_1TeeStream>`;

	// global functions

	void :ref:`substitute<doxid-d0/d02/namespaceCodeGenerator_1aa4e4834ac812b0cb4663112e4bd49eb2>`(std::string& s, const std::string& trg, const std::string& rep);
	bool :ref:`regexVarSubstitute<doxid-d0/d02/namespaceCodeGenerator_1a80838daf20cefe142f4af7ec2361bfd5>`(std::string& s, const std::string& trg, const std::string& rep);
	bool :ref:`regexFuncSubstitute<doxid-d0/d02/namespaceCodeGenerator_1a6f8f82386dbf754701a2c5968614594f>`(std::string& s, const std::string& trg, const std::string& rep);

	void :ref:`functionSubstitute<doxid-d0/d02/namespaceCodeGenerator_1a7308be23a7721f3913894290bcdd7831>`(
		std::string& code,
		const std::string& funcName,
		unsigned int numParams,
		const std::string& replaceFuncTemplate
		);

	template <typename NameIter>
	void :ref:`name_substitutions<doxid-d0/d02/namespaceCodeGenerator_1af0d563118cb96804450e398824d9dcbb>`(
		std::string& code,
		const std::string& prefix,
		NameIter namesBegin,
		NameIter namesEnd,
		const std::string& postfix = "",
		const std::string& ext = ""
		);

	void :ref:`name_substitutions<doxid-d0/d02/namespaceCodeGenerator_1ad9f131d82605f49225e48d8e1a92be1c>`(
		std::string& code,
		const std::string& prefix,
		const std::vector<std::string>& names,
		const std::string& postfix = "",
		const std::string& ext = ""
		);

	template <
		class T,
		typename std::enable_if< std::is_floating_point< T >::value >::type * = nullptr
		>
	void :ref:`writePreciseString<doxid-d0/d02/namespaceCodeGenerator_1ab6085ea1d46a8959cf26df18c9675b61>`(
		std::ostream& os,
		T value
		);

	template <
		class T,
		typename std::enable_if< std::is_floating_point< T >::value >::type * = nullptr
		>
	std::string :ref:`writePreciseString<doxid-d0/d02/namespaceCodeGenerator_1ada66f2dbbdc1120868dcdd7e994d467c>`(T value);

	template <typename NameIter>
	void :ref:`value_substitutions<doxid-d0/d02/namespaceCodeGenerator_1aa603c9ae203c0e36b5555755b80c61cc>`(
		std::string& code,
		NameIter namesBegin,
		NameIter namesEnd,
		const std::vector<double>& values,
		const std::string& ext = ""
		);

	void :ref:`value_substitutions<doxid-d0/d02/namespaceCodeGenerator_1aca94af4afc7c80b1f16436c683d3646c>`(
		std::string& code,
		const std::vector<std::string>& names,
		const std::vector<double>& values,
		const std::string& ext = ""
		);

	std::string :ref:`ensureFtype<doxid-d0/d02/namespaceCodeGenerator_1a22199ae12a7875826210e2f57ee0b1ee>`(const std::string& oldcode, const std::string& type);
	void :ref:`checkUnreplacedVariables<doxid-d0/d02/namespaceCodeGenerator_1a10af4d74175240a715403e6b5f2103cf>`(const std::string& code, const std::string& codeName);

	void :ref:`preNeuronSubstitutionsInSynapticCode<doxid-d0/d02/namespaceCodeGenerator_1a50a3d09c47901d799d7732f4ed6d2f58>`(
		std::string& wCode,
		const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
		const std::string& offset,
		const std::string& axonalDelayOffset,
		const std::string& postIdx,
		const std::string& devPrefix,
		const std::string& preVarPrefix = "",
		const std::string& preVarSuffix = ""
		);

	void :ref:`postNeuronSubstitutionsInSynapticCode<doxid-d0/d02/namespaceCodeGenerator_1a1d9105793919659e85634fe4a3d94900>`(
		std::string& wCode,
		const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
		const std::string& offset,
		const std::string& backPropDelayOffset,
		const std::string& preIdx,
		const std::string& devPrefix,
		const std::string& postVarPrefix = "",
		const std::string& postVarSuffix = ""
		);

	void :ref:`neuronSubstitutionsInSynapticCode<doxid-d0/d02/namespaceCodeGenerator_1ab792dd23b63e89e1fce0947a3c2aaba7>`(
		std::string& wCode,
		const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
		const std::string& preIdx,
		const std::string& postIdx,
		const std::string& devPrefix,
		double dt,
		const std::string& preVarPrefix = "",
		const std::string& preVarSuffix = "",
		const std::string& postVarPrefix = "",
		const std::string& postVarSuffix = ""
		);

	:ref:`GENN_EXPORT<doxid-d1/d8e/gennExport_8h_1a8224d44517aa3e4a332fbd342364f2e7>` std::ostream& :target:`operator <<<doxid-d0/d02/namespaceCodeGenerator_1a950a9d11a55f077e3a144728fc0a7ff0>` (
		std::ostream& s,
		const :ref:`CodeStream::OB<doxid-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB>`& ob
		);

	:ref:`GENN_EXPORT<doxid-d1/d8e/gennExport_8h_1a8224d44517aa3e4a332fbd342364f2e7>` std::ostream& :target:`operator <<<doxid-d0/d02/namespaceCodeGenerator_1aa9430a13006943db52167ab768d63e54>` (
		std::ostream& s,
		const :ref:`CodeStream::CB<doxid-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB>`& cb
		);

	:ref:`GENN_EXPORT<doxid-d1/d8e/gennExport_8h_1a8224d44517aa3e4a332fbd342364f2e7>` std::vector<std::string> :target:`generateAll<doxid-d0/d02/namespaceCodeGenerator_1a702f15415adec44845f9420eb485dd6a>`(
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend,
		const filesystem::path& outputPath,
		bool standaloneModules = false
		);

	void :target:`generateInit<doxid-d0/d02/namespaceCodeGenerator_1a55adab853949f40aae9d01043872cad0>`(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend,
		bool standaloneModules
		);

	void :ref:`GENN_EXPORT<doxid-d1/d8e/gennExport_8h_1a8224d44517aa3e4a332fbd342364f2e7>` :target:`generateMakefile<doxid-d0/d02/namespaceCodeGenerator_1a48a0efb8eb40969e45c54f39c6a6aa8d>`(
		std::ostream& os,
		const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend,
		const std::vector<std::string>& moduleNames
		);

	void :ref:`GENN_EXPORT<doxid-d1/d8e/gennExport_8h_1a8224d44517aa3e4a332fbd342364f2e7>` :ref:`generateMPI<doxid-d0/d02/namespaceCodeGenerator_1ab064c9ce4812db4d3616b89c9c292ec2>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model, const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend, bool standaloneModules);

	void :ref:`GENN_EXPORT<doxid-d1/d8e/gennExport_8h_1a8224d44517aa3e4a332fbd342364f2e7>` :target:`generateMSBuild<doxid-d0/d02/namespaceCodeGenerator_1a09921f5dc44e788c7060f559b5469802>`(
		std::ostream& os,
		const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend,
		const std::string& projectGUID,
		const std::vector<std::string>& moduleNames
		);

	void :target:`generateNeuronUpdate<doxid-d0/d02/namespaceCodeGenerator_1a722e720130ce81d3610c4bffa00b957d>`(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend,
		bool standaloneModules
		);

	:ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :target:`generateRunner<doxid-d0/d02/namespaceCodeGenerator_1a4b99dd706d4c435e17e522cdf302cc0d>`(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend,
		int localHostID
		);

	void :target:`generateSupportCode<doxid-d0/d02/namespaceCodeGenerator_1a5b65889dde61b596c31bfc428f1bf91c>`(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model
		);

	void :target:`generateSynapseUpdate<doxid-d0/d02/namespaceCodeGenerator_1a7737e92de770ca0a57a7c3a642c329e0>`(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend,
		bool standaloneModules
		);

	} // namespace CodeGenerator
.. _details-d0/d02/namespaceCodeGenerator:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Helper class for generating code - automatically inserts brackets, indents etc.

Based heavily on: `https://stackoverflow.com/questions/15053753/writing-a-manipulator-for-a-custom-stream-class <https://stackoverflow.com/questions/15053753/writing-a-manipulator-for-a-custom-stream-class>`__

Global Functions
----------------

.. index:: pair: function; substitute
.. _doxid-d0/d02/namespaceCodeGenerator_1aa4e4834ac812b0cb4663112e4bd49eb2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void substitute(std::string& s, const std::string& trg, const std::string& rep)

Tool for substituting strings in the neuron code strings or other templates.

.. index:: pair: function; regexVarSubstitute
.. _doxid-d0/d02/namespaceCodeGenerator_1a80838daf20cefe142f4af7ec2361bfd5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool regexVarSubstitute(
		std::string& s,
		const std::string& trg,
		const std::string& rep
		)

Tool for substituting variable names in the neuron code strings or other templates using regular expressions.

.. index:: pair: function; regexFuncSubstitute
.. _doxid-d0/d02/namespaceCodeGenerator_1a6f8f82386dbf754701a2c5968614594f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool regexFuncSubstitute(
		std::string& s,
		const std::string& trg,
		const std::string& rep
		)

Tool for substituting function names in the neuron code strings or other templates using regular expressions.

.. index:: pair: function; functionSubstitute
.. _doxid-d0/d02/namespaceCodeGenerator_1a7308be23a7721f3913894290bcdd7831:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void functionSubstitute(
		std::string& code,
		const std::string& funcName,
		unsigned int numParams,
		const std::string& replaceFuncTemplate
		)

This function substitutes function calls in the form:

$(functionName, parameter1, param2Function(0.12, "string"))

with replacement templates in the form:

actualFunction(CONSTANT, $(0), $(1))

.. index:: pair: function; name_substitutions
.. _doxid-d0/d02/namespaceCodeGenerator_1af0d563118cb96804450e398824d9dcbb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename NameIter>
	void name_substitutions(
		std::string& code,
		const std::string& prefix,
		NameIter namesBegin,
		NameIter namesEnd,
		const std::string& postfix = "",
		const std::string& ext = ""
		)

This function performs a list of name substitutions for variables in code snippets.

.. index:: pair: function; name_substitutions
.. _doxid-d0/d02/namespaceCodeGenerator_1ad9f131d82605f49225e48d8e1a92be1c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void name_substitutions(
		std::string& code,
		const std::string& prefix,
		const std::vector<std::string>& names,
		const std::string& postfix = "",
		const std::string& ext = ""
		)

This function performs a list of name substitutions for variables in code snippets.

.. index:: pair: function; writePreciseString
.. _doxid-d0/d02/namespaceCodeGenerator_1ab6085ea1d46a8959cf26df18c9675b61:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <
		class T,
		typename std::enable_if< std::is_floating_point< T >::value >::type * = nullptr
		>
	void writePreciseString(
		std::ostream& os,
		T value
		)

This function writes a floating point value to a stream -setting the precision so no digits are lost.

.. index:: pair: function; writePreciseString
.. _doxid-d0/d02/namespaceCodeGenerator_1ada66f2dbbdc1120868dcdd7e994d467c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <
		class T,
		typename std::enable_if< std::is_floating_point< T >::value >::type * = nullptr
		>
	std::string writePreciseString(T value)

This function writes a floating point value to a string - setting the precision so no digits are lost.

.. index:: pair: function; value_substitutions
.. _doxid-d0/d02/namespaceCodeGenerator_1aa603c9ae203c0e36b5555755b80c61cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename NameIter>
	void value_substitutions(
		std::string& code,
		NameIter namesBegin,
		NameIter namesEnd,
		const std::vector<double>& values,
		const std::string& ext = ""
		)

This function performs a list of value substitutions for parameters in code snippets.

.. index:: pair: function; value_substitutions
.. _doxid-d0/d02/namespaceCodeGenerator_1aca94af4afc7c80b1f16436c683d3646c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void value_substitutions(
		std::string& code,
		const std::vector<std::string>& names,
		const std::vector<double>& values,
		const std::string& ext = ""
		)

This function performs a list of value substitutions for parameters in code snippets.

.. index:: pair: function; ensureFtype
.. _doxid-d0/d02/namespaceCodeGenerator_1a22199ae12a7875826210e2f57ee0b1ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::string ensureFtype(const std::string& oldcode, const std::string& type)

This function implements a parser that converts any floating point constant in a code snippet to a floating point constant with an explicit precision (by appending "f" or removing it).

.. index:: pair: function; checkUnreplacedVariables
.. _doxid-d0/d02/namespaceCodeGenerator_1a10af4d74175240a715403e6b5f2103cf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void checkUnreplacedVariables(
		const std::string& code,
		const std::string& codeName
		)

This function checks for unknown variable definitions and returns a gennError if any are found.

.. index:: pair: function; preNeuronSubstitutionsInSynapticCode
.. _doxid-d0/d02/namespaceCodeGenerator_1a50a3d09c47901d799d7732f4ed6d2f58:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void preNeuronSubstitutionsInSynapticCode(
		std::string& wCode,
		const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
		const std::string& offset,
		const std::string& axonalDelayOffset,
		const std::string& postIdx,
		const std::string& devPrefix,
		const std::string& preVarPrefix = "",
		const std::string& preVarSuffix = ""
		)

suffix to be used for presynaptic variable accesses - typically combined with prefix to wrap in function call such as \__ldg(&XXX)

Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.

.. index:: pair: function; postNeuronSubstitutionsInSynapticCode
.. _doxid-d0/d02/namespaceCodeGenerator_1a1d9105793919659e85634fe4a3d94900:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void postNeuronSubstitutionsInSynapticCode(
		std::string& wCode,
		const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
		const std::string& offset,
		const std::string& backPropDelayOffset,
		const std::string& preIdx,
		const std::string& devPrefix,
		const std::string& postVarPrefix = "",
		const std::string& postVarSuffix = ""
		)

suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as \__ldg(&XXX)

.. index:: pair: function; neuronSubstitutionsInSynapticCode
.. _doxid-d0/d02/namespaceCodeGenerator_1ab792dd23b63e89e1fce0947a3c2aaba7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void neuronSubstitutionsInSynapticCode(
		std::string& wCode,
		const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
		const std::string& preIdx,
		const std::string& postIdx,
		const std::string& devPrefix,
		double dt,
		const std::string& preVarPrefix = "",
		const std::string& preVarSuffix = "",
		const std::string& postVarPrefix = "",
		const std::string& postVarSuffix = ""
		)

Function for performing the code and value substitutions necessary to insert neuron related variables, parameters, and extraGlobal parameters into synaptic code.

suffix to be used for postsynaptic variable accesses - typically combined with prefix to wrap in function call such as \__ldg(&XXX)

.. index:: pair: function; generateMPI
.. _doxid-d0/d02/namespaceCodeGenerator_1ab064c9ce4812db4d3616b89c9c292ec2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void :ref:`GENN_EXPORT<doxid-d1/d8e/gennExport_8h_1a8224d44517aa3e4a332fbd342364f2e7>` generateMPI(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model, const :ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`& backend, bool standaloneModules)

A function that generates predominantly MPI infrastructure code.

In this function MPI infrastructure code are generated, including: MPI send and receive functions.

