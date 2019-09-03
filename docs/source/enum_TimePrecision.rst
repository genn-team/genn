.. index:: pair: enum; TimePrecision
.. _doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8f:

enum TimePrecision
==================

Overview
~~~~~~~~

Precision to use for variables which store time. :ref:`More...<details-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <modelSpec.h>

	enum TimePrecision
	{
	    :ref:`DEFAULT<doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8fa5b39c8b553c821e7cddc6da64b5bd2ee>`,
	    :ref:`FLOAT<doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8fae738c26bf4ce1037fa81b039a915cbf6>`,
	    :ref:`DOUBLE<doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8fafd3e4ece78a7d422280d5ed379482229>`,
	};

.. _details-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Precision to use for variables which store time.

Enum Values
-----------

.. index:: pair: enumvalue; DEFAULT
.. _doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8fa5b39c8b553c821e7cddc6da64b5bd2ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	DEFAULT

Time uses default model precision.

.. index:: pair: enumvalue; FLOAT
.. _doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8fae738c26bf4ce1037fa81b039a915cbf6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	FLOAT

Time uses single precision - not suitable for long simulations.

.. index:: pair: enumvalue; DOUBLE
.. _doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8fafd3e4ece78a7d422280d5ed379482229:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	DOUBLE

Time uses double precision - may reduce performance.

