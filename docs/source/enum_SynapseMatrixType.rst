.. index:: pair: enum; SynapseMatrixType
.. _doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9c:

enum SynapseMatrixType
======================



.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <synapseMatrixType.h>

	enum SynapseMatrixType
	{
	    :target:`DENSE_GLOBALG<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca0103dab4be5e9b66601b43a52ffa00f0>`                  = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
	    :target:`DENSE_GLOBALG_INDIVIDUAL_PSM<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca05bf2ba82e234d9d8ba1b92b6287945e>`   = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
	    :target:`DENSE_INDIVIDUALG<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9cac125fea63eb10ca9b8951ddbe787d7ce>`              = static_cast<unsigned int>(SynapseMatrixConnectivity::DENSE) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
	    :target:`BITMASK_GLOBALG<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca1655cb54ae8edd2462977f30072f8bf8>`                = static_cast<unsigned int>(SynapseMatrixConnectivity::BITMASK) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
	    :target:`BITMASK_GLOBALG_INDIVIDUAL_PSM<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca1afc3ca441931cf66047766d6a135ff4>` = static_cast<unsigned int>(SynapseMatrixConnectivity::BITMASK) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
	    :target:`SPARSE_GLOBALG<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>`                 = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL),
	    :target:`SPARSE_GLOBALG_INDIVIDUAL_PSM<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca4caebb15c1a09f263b6f223241bde1ac>`  = static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixWeight::GLOBAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
	    :target:`SPARSE_INDIVIDUALG<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9cae7658b74f700d52b421afc540c892d2e>`             =  static_cast<unsigned int>(SynapseMatrixConnectivity::SPARSE) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL) | static_cast<unsigned int>(SynapseMatrixWeight::INDIVIDUAL_PSM),
	};

