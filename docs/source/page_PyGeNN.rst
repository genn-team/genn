.. index:: pair: page; Python interface (PyGeNN)
.. _doxid-d0/d81/PyGeNN:

Python interface (PyGeNN)
=========================

As well as being able to build GeNN models and user code directly from C++, you can also access all GeNN features from Python. The ``:ref:`pygenn.genn_model.GeNNModel <doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>``` class provides a thin wrapper around ``:ref:`ModelSpec <doxid-da/dfd/classModelSpec>``` as well as providing support for loading and running simulations; and accessing their state. ``:ref:`SynapseGroup <doxid-dc/dfa/classSynapseGroup>```, ``:ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>``` and ``:ref:`CurrentSource <doxid-d1/d48/classCurrentSource>``` are similarly wrapped by the ``:ref:`pygenn.genn_groups.SynapseGroup <doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>```, ``:ref:`pygenn.genn_groups.NeuronGroup <doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>``` and ``:ref:`pygenn.genn_groups.CurrentSource <doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>``` classes respectively.

PyGeNN can be built from source on Windows, Mac and Linux following the instructions in the README file in the pygenn directory of the GeNN repository. However, if you have a relatively recent version of Python and CUDA, we recommend that you instead downloading a suitable 'wheel' from our releases page. These can then be installed using e.g. ``pip install cuda10-pygenn-0.2-cp27-cp27mu-linux_x86_64.whl`` for a Linux system with CUDA 10 and Python 2.7. On Windows we recommend using the Python 3 version of `Anaconda <https://www.anaconda.com/distribution/>`__.

The following example shows how PyGeNN can be easily interfaced with standard Python packages such as numpy and matplotlib to plot 4 different Izhikevich neuron regimes:

.. ref-code-block:: cpp

	import numpy as np
	import matplotlib.pyplot as plt
	from :ref:`pygenn.genn_model <doxid-de/d6e/namespacepygenn_1_1genn__model>` import GeNNModel
	
	# Create a single-precision GeNN model
	model = GeNNModel("float", "pygenn")
	
	# Set simulation timestep to 0.1ms
	model.dT = 0.1
	
	# Initialise IzhikevichVariable parameters - arrays will be automatically uploaded
	izk_init = {"V": -65.0,
	            "U": -20.0,
	            "a": [0.02,     0.1,    0.02,   0.02],
	            "b": [0.2,      0.2,    0.2,    0.2],
	            "c": [-65.0,    -65.0,  -50.0,  -55.0],
	            "d": [8.0,      2.0,    2.0,    4.0]}
	
	# Add neuron populations and current source to model
	pop = model.add_neuron_population("Neurons", 4, "IzhikevichVariable", {}, izk_init)
	model.add_current_source("CurrentSource", "DC", "Neurons", {"amp": 10.0}, {})
	
	# Build and load model
	model.build()
	model.load()
	
	# Create a numpy view to efficiently access the membrane voltage from Python
	voltage_view = pop.vars["V"].view
	
	# Simulate
	v = None
	while model.t < 200.0:
	    model.step_time()
	    model.pull_state_from_device("Neurons")
	    v = np.copy(voltage_view) if v is None else np.vstack((v, voltage_view))
	
	# Create plot
	figure, axes = plt.subplots(4, sharex=True)
	
	# Plot voltages
	for i, t in enumerate(["RS", "FS", "CH", "IB"]):
	    axes[i].set_title(t)
	    axes[i].set_ylabel("V [mV]")
	    axes[i].plot(np.arange(0.0, 200.0, 0.1), v[:,i])
	
	axes[-1].set_xlabel("Time [ms]")
	
	# Show plot
	plt.show()

:ref:`Previous <doxid-d3/d0c/brian2genn>` \| :ref:`Top <doxid-d0/d81/PyGeNN>` \| :ref:`Next <doxid-df/ddb/ReleaseNotes>`

