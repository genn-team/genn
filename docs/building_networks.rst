.. py:currentmodule:: pygenn

=================
Building networks
=================
---------
The model
---------
A network model is defined as follows:
A :class:`.GeNNModel` must be created with a name and a default precision (see \ref floatPrecision)}:

..  code-block:: python

    model = GeNNModel("float", "YourModelName")

By default the model will use a hardware-accelerated code-generation backend if it is available. 
However, this can be overriden using the `backend` keyword argument. 
For example, the single-threaded CPU backend could be manually selected with:

..  code-block:: python

    model = GeNNModel("float", "YourModelName", 
                      backend="single_threaded_cpu")

Additionally, any preferences exposed by the backend can be configured here. 
For example, the CUDA backend allows you to select which CUDA device to use via the manual_device_id

-----------
Populations
-----------

Parameters
----------
Parameters are homogeneous across an entire population and 


Variables
----------
Variables can be initialised in many ways.
By using the GPU to fill them with a constant value:

..  code-block:: python

    ini = {"m": 0.0529324, ...}

by copying in a sequence of values from Python:

..  code-block:: python

    ini = {"m": np.arange(400.0), ...}

or by specifying a variable initialisation snippet which can itself be configured with parameters:

..  code-block:: python

    params = {"mean": 0.05, "sd": 0.01}
    ini = {"m": pygenn.init_var("Normal", params), ...}

where built in snippets, included in the :mod:`.init_var_snippets` module, can be selected by name like "Normal".
However, like many other parts of GeNN, you can easily create your own variable initialisation snippets as described in TODO.

Variables references
--------------------
Advanced!

Neuron populations
------------------
Neuron populations contain a number of neurons with the same model and are added using:

.. automethod:: pygenn.GeNNModel.add_neuron_population

The user may add as many neuron populations as the model necessitates.
They must all have unique names. The possible values for the arguments,
predefined models and their parameters and initial values are detailed


Synapse populations
-------------------
Synapse populations connect two neuron populations via synapses:

.. automethod:: pygenn.GeNNModel.add_synapse_population

Current sources
---------------
Current sources 

.. automethod:: pygenn.GeNNModel.add_current_source

Custom updates
--------------
Current sources 

.. automethod:: pygenn.GeNNModel.add_custom_update

Custom connectivity updates
---------------------------
Current sources 

.. automethod:: pygenn.GeNNModel.add_custom_connectivity_update
