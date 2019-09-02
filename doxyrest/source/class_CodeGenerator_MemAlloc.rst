.. index:: pair: class; CodeGenerator::MemAlloc
.. _doxid-d2/d06/classCodeGenerator_1_1MemAlloc:

class CodeGenerator::MemAlloc
=============================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <backendBase.h>
	
	class MemAlloc
	{
	public:
		// methods
	
		size_t :target:`getHostBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a6c999467803d3fca57d26898cb4dd1a3>`() const;
		size_t :target:`getDeviceBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a30883116724d4475b7999f52b9ccefa6>`() const;
		size_t :target:`getZeroCopyBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1abc2602688ece395a2798a1f72eb9b810>`() const;
		size_t :target:`getHostMBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a7087a0f011544619f53b7a0a4efa60c9>`() const;
		size_t :target:`getDeviceMBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a8b186723630854ea70c1f6649cbcbb32>`() const;
		size_t :target:`getZeroCopyMBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a2faa6b6734e92998dd62a7465de56c70>`() const;
		MemAlloc& :target:`operator +=<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a97de8e89cbdb26df2cfbc0c45bc46324>` (const MemAlloc& rhs);
		static MemAlloc :target:`zero<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a855d60ed10ef7b083bd0efd360ad4184>`();
		static MemAlloc :target:`host<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a6f6fb900415ced795d5cb73eb09582c4>`(size_t hostBytes);
		static MemAlloc :target:`device<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a07e73ba804957f36166395d41f89640c>`(size_t deviceBytes);
		static MemAlloc :target:`zeroCopy<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1abb8977011232ea837e81cef7b1b33cbc>`(size_t zeroCopyBytes);
	};
