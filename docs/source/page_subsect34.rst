.. index:: pair: page; Synaptic matrix types
.. _doxid-d5/d39/subsect34:

Synaptic matrix types
=====================

Synaptic matrix types are made up of two components: SynapseMatrixConnectivity and SynapseMatrixWeight. SynapseMatrixConnectivity defines what data structure is used to store the synaptic matrix:

* :ref:`SynapseMatrixConnectivity::DENSE <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0706fbbd929bd8abc4de386c53d439ff>` stores synaptic matrices as a dense matrix. Large dense matrices require a large amount of memory and if they contain a lot of zeros it may be inefficient.

* :ref:`SynapseMatrixConnectivity::SPARSE <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0459833ba9cad7cfd7bbfe10d7bbbe6e>` stores synaptic matrices in a(padded) 'ragged array' format. In general, this is less efficient to traverse using a GPU than the dense matrix format but does result in significant memory savings for large matrices. Ragged matrix connectivity is stored using several variables whose names, like state variables, have the name of the synapse population appended to them:
  
  #. ``const unsigned int maxRowLength`` : a constant set via the ``:ref:`SynapseGroup::setMaxConnections <doxid-dc/dfa/classSynapseGroup_1aab6b2fb0ad30189bc11ee3dd7d48dbb2>``` method which specifies the maximum number of connections in any given row (this is the width the structure is padded to).
  
  #. ``unsigned int *rowLength`` (sized to number of presynaptic neurons): actual length of the row of connections associated with each presynaptic neuron
  
  #. ``unsigned int *ind`` (sized to ``maxRowLength * number of presynaptic neurons``): Indices of corresponding postsynaptic neurons concatenated for each presynaptic neuron. For example, consider a network of two presynaptic neurons connected to three postsynaptic neurons: 0th presynaptic neuron connected to 1st and 2nd postsynaptic neurons, the 1st presynaptic neuron connected only to the 0th neuron. The struct RaggedProjection should have these members, with indexing from 0 (where X represents a padding value):
     
     .. ref-code-block:: cpp
     
     	maxRowLength = 2
     	ind = [1 2 0 X]
     	rowLength = [2 1]
     
     Weight update model variables associated with the sparsely connected synaptic population will be kept in an array using the same indexing as ind. For example, a variable caled ``g`` will be kept in an array such as: ``g=`` [g_Pre0-Post1 g_pre0-post2 g_pre1-post0 X]

* :ref:`SynapseMatrixConnectivity::BITMASK <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0287e103671bf22378919a64d4b70699>` is an alternative sparse matrix implementation where which synapses within the matrix are present is specified as a binary array (see :ref:`Insect olfaction model <doxid-d9/d61/Examples_1ex_mbody>`). This structure is somewhat less efficient than the ``:ref:`SynapseMatrixConnectivity::SPARSE <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0459833ba9cad7cfd7bbfe10d7bbbe6e>``` and ``SynapseMatrixConnectivity::RAGGED`` formats and doesn't allow individual weights per synapse. However it does require the smallest amount of GPU memory for large networks.

Furthermore the SynapseMatrixWeight defines how

* :ref:`SynapseMatrixWeight::INDIVIDUAL <doxid-dd/dd5/synapseMatrixType_8h_1a3c0f0120d3cb9e81daea1d2afa7fbe1fa938873bbf7fe69b2f3836e6103f2a323>` allows each individual synapse to have unique weight update model variables. Their values must be initialised at runtime and, if running on the GPU, copied across from the user side code, using the ``pushXXXXXStateToDevice`` function, where XXXX is the name of the synapse population.

* :ref:`SynapseMatrixWeight::INDIVIDUAL_PSM <doxid-dd/dd5/synapseMatrixType_8h_1a3c0f0120d3cb9e81daea1d2afa7fbe1faf92bc2c3cbbf79265bfd8deb87b087fa>` allows each postsynapic neuron to have unique post synaptic model variables. Their values must be initialised at runtime and, if running on the GPU, copied across from the user side code, using the ``pushXXXXXStateToDevice`` function, where XXXX is the name of the synapse population.

* :ref:`SynapseMatrixWeight::GLOBAL <doxid-dd/dd5/synapseMatrixType_8h_1a3c0f0120d3cb9e81daea1d2afa7fbe1fa6eecfba72d12922ee1dead07a0ef3334>` saves memory by only maintaining one copy of the weight update model variables. This is automatically initialized to the initial value passed to :ref:`ModelSpec::addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`.

Only certain combinations of SynapseMatrixConnectivity and SynapseMatrixWeight are sensible therefore, to reduce confusion, the SynapseMatrixType enumeration defines the following options which can be passed to :ref:`ModelSpec::addSynapsePopulation <doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>` :

* :ref:`SynapseMatrixType::SPARSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca14329718a99dc337fa3bd33b9104d75d>`

* :ref:`SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca4caebb15c1a09f263b6f223241bde1ac>`

* :ref:`SynapseMatrixType::SPARSE_INDIVIDUALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9cae7658b74f700d52b421afc540c892d2e>`

* :ref:`SynapseMatrixType::DENSE_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca0103dab4be5e9b66601b43a52ffa00f0>`

* :ref:`SynapseMatrixType::DENSE_GLOBALG_INDIVIDUAL_PSM <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca05bf2ba82e234d9d8ba1b92b6287945e>`

* :ref:`SynapseMatrixType::DENSE_INDIVIDUALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9cac125fea63eb10ca9b8951ddbe787d7ce>`

* :ref:`SynapseMatrixType::BITMASK_GLOBALG <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca1655cb54ae8edd2462977f30072f8bf8>`

* :ref:`SynapseMatrixType::BITMASK_GLOBALG_INDIVIDUAL_PSM <doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9ca1afc3ca441931cf66047766d6a135ff4>`

:ref:`Previous <doxid-d0/d1e/sectCurrentSourceModels>` \| :ref:`Top <doxid-dc/d05/UserManual>` \| :ref:`Next <doxid-d4/dc6/sectVariableInitialisation>`

