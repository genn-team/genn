.. index:: pair: page; Brian interface (Brian2GeNN)
.. _doxid-d3/d0c/brian2genn:

Brian interface (Brian2GeNN)
============================

GeNN can simulate models written for the `Brian simulator <http://brian2.readthedocs.io>`__ via the `Brian2GeNN <http://brian2genn.readthedocs.io>`__ interface :ref:`brian2genn2018 <doxid-d0/de3/citelist_1CITEREF_brian2genn2018>`. The easiest way to install everything needed is to install the `Anaconda <https://www.anaconda.com/download>`__ or `Miniconda <https://conda.io/miniconda.html>`__ Python distribution and then follow the `instructions to install Brian2GeNN <http://brian2genn.readthedocs.io/en/latest/introduction/index.html#installing-the-brian2genn-interface>`__ with the conda package manager. When Brian2GeNN is installed in this way, it comes with a bundled version of GeNN and no further configuration is required. In all other cases (e.g. an installation from source), the path to GeNN and the CUDA libraries has to be configured via the ``GENN_PATH`` and ``CUDA_PATH`` environment variables as described in :ref:`Installation <doxid-d8/d99/Installation>` or via the ``devices.genn.path`` and ``devices.genn.cuda_path`` `Brian preferences <http://brian2.readthedocs.io/en/stable/advanced/preferences.html>`__.

To use GeNN to simulate a Brian script, import the ``brian2genn`` package and switch Brian to the ``genn`` device. As an example, the following Python script will simulate Leaky-integrate-and-fire neurons with varying input currents to construct an f/I curve:

.. ref-code-block:: cpp

	from brian2 import *
	import brian2genn
	set_device('genn')
	
	n = 1000
	duration = 1*second
	tau = 10*ms
	eqs = '''
	dv/dt = (v0 - v) / tau : volt (unless refractory)
	v0 : volt
	'''
	group = :ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>`(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
	                    refractory=5*ms, method='exact')
	group.v = 0*mV
	group.v0 = '20*mV * i / (n-1)'
	monitor = SpikeMonitor(group)
	
	run(duration)

Of course, your simulation should be more complex than the example above to actually benefit from the performance gains of using a GPU via GeNN. :ref:`Previous <doxid-d2/dba/SpineML>` \| :ref:`Top <doxid-d3/d0c/brian2genn>` \| :ref:`Next <doxid-d0/d81/PyGeNN>`

