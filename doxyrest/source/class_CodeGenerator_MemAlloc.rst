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
	
		size_t :target:`getHostBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1aa2bdc12fe981affd96a1f752c18f913e>`() const;
		size_t :target:`getDeviceBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1ac135d06724cf453a94470111c2560845>`() const;
		size_t :target:`getZeroCopyBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1aa1ce0f74332a9e2ad31864ffee84aa58>`() const;
		size_t :target:`getHostMBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a2546ad8fec9d3306a6f5f6a74031cf2d>`() const;
		size_t :target:`getDeviceMBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a1939c0a64e4fc3a84d8e744c6c4f6cfe>`() const;
		size_t :target:`getZeroCopyMBytes<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1aabd40130f9f70de1134492d29f39cd35>`() const;
		MemAlloc& :target:`operator +=<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a97de8e89cbdb26df2cfbc0c45bc46324>` (const MemAlloc& rhs);
		static MemAlloc :target:`zero<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a855d60ed10ef7b083bd0efd360ad4184>`();
		static MemAlloc :target:`host<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a6f6fb900415ced795d5cb73eb09582c4>`(size_t hostBytes);
		static MemAlloc :target:`device<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1a07e73ba804957f36166395d41f89640c>`(size_t deviceBytes);
		static MemAlloc :target:`zeroCopy<doxid-d2/d06/classCodeGenerator_1_1MemAlloc_1abb8977011232ea837e81cef7b1b33cbc>`(size_t zeroCopyBytes);
	};
