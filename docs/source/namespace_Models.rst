.. index:: pair: namespace; Models
.. _doxid-dd/d20/namespaceModels:

namespace Models
================

.. toctree::
	:hidden:

	class_Models_Base.rst
	class_Models_VarInit.rst
	class_Models_VarInitContainerBase.rst
	class_Models_VarInitContainerBase-2.rst

Class used to bind together everything required to initialise a variable:

#. A pointer to a variable initialisation snippet

#. The parameters required to control the variable initialisation snippet


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace Models {

	// classes

	class :ref:`Base<doxid-d6/d97/classModels_1_1Base>`;
	class :ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`;

	template <>
	class :ref:`VarInitContainerBase<0><doxid-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4>`;

	template <size_t NumVars>
	class :ref:`VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`;

	} // namespace Models
