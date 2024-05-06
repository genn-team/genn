.. py:currentmodule:: pygenn

.. _`section-building-networks`:

=================
Building networks
=================
---------
The model
---------
A network model is defined as follows:
A :class:`.GeNNModel` must be created with a default precision (see \ref floatPrecision) and a name:

..  code-block:: python

    model = GeNNModel("float", "YourModelName")

By default the model will use a hardware-accelerated code-generation backend if it is available. 
However, this can be overriden using the `backend` keyword argument. 
For example, the single-threaded CPU backend could be manually selected with:

..  code-block:: python

    model = GeNNModel("float", "YourModelName", 
                      backend="single_threaded_cpu")

When running models on a GPU, smaller models may not fully occupy the device. In some scenarios such as gradient-based training and parameter sweeping,
this can be overcome by running multiple copies of the same model at the same time (batching in Machine Learning speak).
Batching can be enabled on a GeNN model with:

..  code-block:: python

    model.batch_size = 512

Parameters and sparse connectivity are shared across all batches. 
Whether state variables are duplicated or shared is controlled by the :class:`.VarAccess` or :class:`.CustomUpdateVarAccess` enumeration 
associated with each variable. Please see TODO for more details.

Additionally, any preferences exposed by the backend can be configured here. 
For example, the CUDA backend allows you to select which CUDA device to use via the manual_device_id:

.. code-block:: python

   model = GeNNModel("float", "YourModelName",
                     backend="cuda", manual_device_id=0)

-----------
Populations
-----------
Populations formalise the concept of groups of neurons or synapses that are functionally related or a practical grouping, e.g. a brain region in a neuroscience model or a layer in a machine learning context.

..  _`section-parameters`:

Parameters
----------
Parameters are initialised to constant numeric values which are homogeneous across an entire population:

..  code-block:: python

    para = {"m": 0.0529324, ...}

They are very efficient to access from models as their values are either hard-coded into the backend code 
or, on the GPU, delivered via high-performance constant cache.
However, they can only be used if all members of the population have the exact same parameter value.
Parameters are always read-only but can be made *dynamic* so they can be changed from the host 
during the simulation by calling :meth:`pygenn.NeuronGroup.set_param_dynamic` on a :class:`.NeuronGroup`, i.e.

..  code-block:: python

    pop.set_param_dynamic("tau")

where ``pop`` is a neuron group returned by :meth:`.GeNNModel.add_neuron_population` or synapse group returned by :meth:`.GeNNModel.add_synapse_population` and "tau" is one of the population's declared parameters.

.. warning::
    Derived parameters will not change if the parameters they rely on are made dynamic and changed at runtime.

Extra Global Parameters
-----------------------
When building more complex models, it is sometimes useful to be able to access arbitarily
sized arrays. In GeNN, these are called Extra Global Parameters (EGPs) and they need
to be manually allocated and initialised before simulating the model. For example, the built-in :func:`.neuron_models.SpikeSourceArray` model has a ``spikeTimes`` EGP
which is used to provide an array of spike times for the spike source to emit. Given two numpy arrays: ``spike_ids`` containing the ids of which neurons spike and
``spike_times`` containing the time at which each spike occurs, a :func:`.neuron_models.SpikeSourceArray` 
model can be configured as follows:

..  code-block:: python
    
    # Calculate start and end index of each neuron's spikes in sorted array
    end_spike = np.cumsum(np.bincount(spike_ids, minlength=100))
    start_spike = np.concatenate(([0], end_spike[0:-1]))

    # Sort events first by neuron id and then 
    # by time and use to order spike times
    spike_times = spike_times[np.lexsort((spike_times, spike_ids))]

    model = GeNNModel("float", "spike_source_array_example")

    ssa = model.add_neuron_population("SSA", 100, "SpikeSourceArray", {}, 
                                      {"startSpike": start_spike, "endSpike": end_spike})
    ssa.extra_global_params["spikeTimes"].set_init_values(spike_times)


..  _`section-variables`:
    
Variables
----------
Variables contain values that are individual to the members of a population and can change over time. They can be initialised in many ways. The initialisation is configured through a Python dictionary that is then passed to :meth:`.GeNNModel.add_neuron_population` or :meth:`.GeNNModel.add_synapse_population` which create the populations.

To initialise variables one can use the backend, e.g. GPU, to fill them with a constant value:

..  code-block:: python

    ini = {"m": 0.0529324, ...}

or copy a sequence of values from Python:

..  code-block:: python

    ini = {"m": np.arange(400.0), ...}

or use a variable initialisation snippet returned by the following function:

.. autofunction:: pygenn.init_var
    :noindex:

The resulting initialisation snippet can then be used in the dictionary in the usual way:

..  code-block:: python

    ini = {"m": init, ...}

..  _`section-variables-references`:

Variables references
--------------------
As well as variables and parameters, various types of models have variable references which are used to reference variables belonging to other populations.
For example, postsynaptic update models can reference variables in the postsynaptic neuron model and custom updates are 'attached' to other populations based on their variable references.

Variable reference can be created to various types of per-neuron variable using:

.. autofunction:: pygenn.create_var_ref
    :noindex:

References can also be created to various types of per-neuron variable owned by synapse groups using: 

.. autofunction:: pygenn.create_psm_var_ref
    :noindex:

.. autofunction:: pygenn.create_wu_pre_var_ref
    :noindex:

.. autofunction:: pygenn.create_wu_post_var_ref
    :noindex:

While references of these types can be used interchangably in the same custom update, as long as all referenced variables have the same delays and belong to populations of the same size, per-synapse weight update model variables must be referenced with slightly different syntax:

.. autofunction:: pygenn.create_wu_var_ref
    :noindex:

These 'weight update variable references' also have the additional feature that they can be used to define a link to a 'transpose' variable:

..  code-block:: python

    wu_transpose_var_ref = {"R": create_wu_var_ref(sg, "g", back_sg, "g")}

where ``back_sg`` is another :class:`.SynapseGroup` with tranposed dimensions to sg i.e. its *postsynaptic* population has the same number of neurons as sg's *presynaptic* population and vice-versa.

After the update has run, any updates made to the 'forward' variable will also be applied to the tranpose variable 

.. note::

    Transposing is currently only possible on variables belonging to synapse groups with :attr:`.SynapseMatrixType.DENSE` connectivity 

Variable locations
------------------
Once you have defined *how* your variables are going to be initialised you need to configure *where* they will be allocated. 
By default memory is allocated for variables on both the GPU and the host.
However, the following alternative 'variable locations' are available:

.. autoclass:: pygenn.VarLocation
    :noindex:

.. note::

    'Zero copy' memory is only supported on newer embedded systems such as
    the Jetson TX1 where there is no physical seperation between GPU and host memory and 
    thus the same physical memory can be shared between them. 

..  _`section-extra-global-parameter-references`:
    
Extra global parameter references
---------------------------------
When building models with complex `Custom updates`_ and `Custom Connectivity updates`_, 
it is often useful to share data stored in extra global parameters between different groups.
Similar to variable references, such links are made using extra global parameter references.
These can be created using:

.. autofunction:: pygenn.create_egp_ref
    :noindex:

.. autofunction:: pygenn.create_psm_egp_ref
    :noindex:

.. autofunction:: pygenn.create_wu_egp_ref
    :noindex:

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
(or other type of input for models that are not current based) and are typically initialised using:

.. autofunction:: pygenn.init_postsynaptic
    :noindex:

GeNN provides a number of different data structures for implementing synaptic connectivity:

.. autoclass:: pygenn.SynapseMatrixType
    :noindex:

:attr:`pygenn.SynapseMatrixType.DENSE` and :attr:`pygenn.SynapseMatrixType.DENSE_PROCEDURAL` 
connectivity can be initialised on the GPU by simply using the variable initialisation snippets described in `Variables`_ 
to initialise the weight update model variables. :attr:`pygenn.SynapseMatrixType.SPARSE`, :attr:`pygenn.SynapseMatrixType.BITMASK` 
and :attr:`pygenn.SynapseMatrixType.PROCEDURAL` synaptic connectivity can be initialised on the GPU using:

.. autofunction:: pygenn.init_sparse_connectivity
    :noindex:

:attr:`pygenn.SynapseMatrixType.TOEPLITZ` can be initialised using:

.. autofunction:: pygenn.init_toeplitz_connectivity
    :noindex:

Finally, with these components in place, a synapse population can be added to the model:

.. automethod:: .GeNNModel.add_synapse_population
    :noindex:

Current sources
---------------
Current sources are added to a model using:

.. automethod:: .GeNNModel.add_current_source
    :noindex:

Custom updates
--------------
The neuron groups, synapse groups and current sources described in previous sections are all updated automatically every timestep.
However, in many types of model, there are also processes that would benefit from GPU acceleration but only need to be triggered occasionally. 
For example, such updates could be used in a classifier to reset the state of neurons after a stimulus has been presented or in a model 
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
