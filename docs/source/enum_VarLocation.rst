.. index:: pair: enum; VarLocation
.. _doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087:

enum VarLocation
================

< Flags defining which memory space variables should be allocated in

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <variableMode.h>

	enum VarLocation
	{
	    :target:`HOST<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087ab9361011891280a44d85b967739cc6a5>`                  = (1 <<0),
	    :target:`DEVICE<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087ae10b6ab6a278644ce40631f62f360b6d>`                = (1 <<1),
	    :target:`ZERO_COPY<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087ae5f97fff9c755d0906f1a4dcdb48ef57>`             = (1 <<2),
	    :target:`HOST_DEVICE<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087aa34547c8e93e562b2c7952c77d426710>`           = HOST | DEVICE,
	    :target:`HOST_DEVICE_ZERO_COPY<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087a42b7a82fbd6d845b0d5c5dbd67846e0d>` = HOST | DEVICE | ZERO_COPY,
	};

