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
Parameters are initialised to constant numeric values which are homogeneous across an entire population:

..  code-block:: python

    ini = {"m": 0.0529324, ...}

They are very efficient to access from models as their values are either hard-coded into the kernels 
or, on the GPU, delivered via high-performance constant cache.
However 

Extra global parameters
-----------------------
TODO

Extra global parameter references
---------------------------------
TODO

Variables
----------
Variables can be initialised in many ways.
By using the GPU to fill them with a constant value:

..  code-block:: python

    ini = {"m": 0.0529324, ...}

by copying in a sequence of values from Python:

..  code-block:: python

    ini = {"m": np.arange(400.0), ...}

or by using a variable initialisation snippet configured using the following function:

.. autofunction:: pygenn.init_var
    :noindex:

and then used in the dictionary the same way:

..  code-block:: python

    ini = {"m": init, ...}

Variables references
--------------------
As well as variables and parameters, various types of model have variable references which are used to reference variables belonging to other populations.
For example, postsynaptic update models can reference variables in the postsynaptic neuron model and custom updates are 'attached' to other populations based on their variable references.

A variable reference called R could be assigned to various types of variable using the following syntax:

..  code-block:: python

    neuron_var_ref =  {"R": pygenn.create_var_ref(ng, "V")}
    current_source_var_ref =  {"R": pygenn.create_var_ref(cs, "V")}
    custom_update_var_ref = {"R": pygenn.create_var_ref(cu, "V")}
    postsynaptic_model_var_ref =  {"R": pygenn.create_psm_var_ref(sg, "V")}
    wu_pre_var_ref =  {"R": pygenn.create_wu_pre_var_ref(sg, "Pre")}
    wu_post_var_ref =  {"R": pygenn.create_wu_post_var_ref(sg, "Post")}

where ``ng`` is a :class:`.NeuronGroup` (as returned by :meth:`.GeNNModel.add_neuron_population`), ``cs`` is a :class:`.CurrentSource` (as returned by :meth:`.GeNNModel.add_current_source`), ``cu`` is a :class:`.CustomUpdate` (as returned by :meth:`.GeNNModel.add_custom_update`) and ``sg`` is a :class:`.SynapseGroup` (as returned by :meth:`.GeNNModel.add_synapse_population`).

While references of these types can be used interchangably in the same custom update, as long as all referenced variables have the same delays and belong to populations of the same size, per-synapse weight update model variables must be referenced with slightly different syntax:

..  code-block:: python

    wu_var_ref = {"R": pygenn.create_wu_var_ref(sg, "g")}
    cu_wu_var_ref = {"R": pygenn.create_wu_var_ref(cu, "g")}

where ``sg`` is a :class:`.SynapseGroup` (as returned by :meth:`.GeNNModel.add_synapse_population`) and ``cu`` is a :class:`.CustomUpdateWU` (as returned by :meth:`.GeNNModel.add_custom_update`) which operates on another synapse group's state variables.

TODO custom connectivity updateBaseHash

These 'weight update variable references' also have the additional feature that they can be used to define a link to a 'transpose' variable:

..  code-block:: python

    wu_transpose_var_ref = {"R": create_wu_var_ref(sg, "g", back_sg, "g")}

where ``back_sg`` is another :class:`.SynapseGroup` with tranposed dimensions to sg i.e. its _postsynaptic_ population has the same number of neurons as sg's _presynaptic_ population and vice-versa.

After the update has run, any updates made to the 'forward' variable will also be applied to the tranpose variable 
[#]_ Tranposing is currently only possible on variables belonging to synapse groups with :attr:`.SynapseMatrixType.DENSE` connectivity [#]_

Variable locations
------------------
Once you have defined *how* your variables are going to be initialised you need to configure *where* they will be allocated. 
By default memory is allocated for variables on both the GPU and the host.
However, the following alternative 'variable locations' are available:

.. autoattribute:: .VarLocation.DEVICE
    :noindex:
.. autoattribute:: .VarLocation.HOST_DEVICE
    :noindex:
.. autoattribute:: .VarLocation.HOST_DEVICE_ZERO_COPY
    :noindex:

Note, 'Zero copy' memory is only supported on newer embedded systems such as
 the Jetson TX1 where there is no physical seperation between GPU and host memory and 
 thus the same physical of memory can be shared between them. 

..
    TODO into enum
    - VarLocation::DEVICE - Variables are only allocated on the GPU, saving memory but meaning that they can't easily be copied to the host - best for internal state variables.
    - VarLocation::HOST_DEVICE - Variables are allocated on both the GPU and the host  - the default.
    - VarLocation::HOST_DEVICE_ZERO_COPY - Variables are allocated as 'zero-copy' memory accessible to the host and GPU - useful on devices such as Jetson TX1 where physical memory is shared between the GPU and CPU.

Neuron populations
------------------
Neuron populations contain a number of neurons with the same model and are added using:

.. automethod:: .GeNNModel.add_neuron_population
    :noindex:

Synapse populations
-------------------
Synapse populations connect two neuron populations via synapses.
Their behaviour is described by a weight update model and a postsynaptic model.
The weight update model defines what kind of dynamics (if any) occurs at 
each synapse and what output they deliver to postsynaptic (and presynaptic) neurons.
Weight update models are typically initialised using:

.. autofunction:: pygenn.init_weight_update
    :noindex:

Postsynaptic models define how synaptic input translates into an input current 
(or other input term for models that are not current based) and are typically initialise using:

.. autofunction:: pygenn.init_postsynaptic
    :noindex:

Additionally synaptic connectivity can be initialised on the GPU using either:

.. autofunction:: pygenn.init_sparse_connectivity
    :noindex:

or:

.. autofunction:: pygenn.init_toeplitz_connectivity
    :noindex:

Finally, with these components in place, a synapse population can be added to the model:

.. automethod:: .GeNNModel.add_synapse_population
    :noindex:

Current sources
---------------
Current sources 

.. automethod:: .GeNNModel.add_current_source
    :noindex:

Custom updates
--------------
The neuron groups, synapse groups and current sources described in previous sections are all updated automatically every timestep.
However, in many types of model, there are also processes that would benefit from GPU acceleration but only need to be triggered occasionally. 
For example, such updates could be used in a classifier to to reset the state of neurons after a stimuli has been presented or in a model 
which uses gradient-based learning to optimize network weights based on gradients accumulated over several timesteps.

Custom updates allows such updates to be described as models, similar to the neuron and synapse models described in the preceding sections. 
The custom update system also provides functionality for efficiently calculating the tranpose of variables associated with synapse groups 
(current only with :attr:`.SynapseMatrixType.DENSE` connectivity). Custom updates are added to a model using:

.. automethod:: .GeNNModel.add_custom_update
    :noindex:

Custom connectivity updates
---------------------------
Like custom update, custom connectivity updates are triggered manually by the user but, rather than 
updating model *variables*, they update model *connectivity* (current only with :attr:`.SynapseMatrixType.SPARSE` connectivity).
Custom connectivity updates are added to a model using:

.. automethod:: .GeNNModel.add_custom_connectivity_update
    :noindex: