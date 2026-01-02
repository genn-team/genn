.. py:currentmodule:: pygenn

===================
Simulating networks
===================
Once you have built a network using the :class:`.GeNNModel` API described in :ref:`section-building-networks` and 
before you can simulate it you first need to launch the GeNN code generator using :meth:`.GeNNModel.build`
and then load the model into memory using :meth:`.GeNNModel.load`. 
Code generation is 'lazy' so if your model hasn't changed, code generation will be almost instantaneous.
If no errors are reported, the simplest simulation looks like the following:

..  code-block:: python

    ...
    model.build()
    model.load()

    while model.timestep < 100:
        model.step_time()

As well as the integer timestep, the current time in ms can be accessed with :attr:`.GeNNModel.t`.
On GPU platforms like CUDA, the above simulation will run asynchronously with the loop 
launching the kernels to simulate each timestep but not synchronising with the CPU at any point.

.. _section-spike-recording:

---------------
Spike recording
---------------
Because recording spikes and spike-like events is a common requirement and their sparse nature can make them inefficient to access,
GeNN has a dedicated events recording system which collects events, emitted over a number of timesteps, in GPU memory before transferring to the host. 
Spike recording can be enabled on chosen neuron groups by setting the :attr:`.NeuronGroup.spike_recording_enabled` and :attr:`.NeuronGroup.spike_event_recording_enabled` properties. 
Memory can then be allocated at runtime for spike recording by using the ``num_recording_timesteps`` keyword argument to :meth:`.GeNNModel.load`.
Spikes can then be copied from the GPU to the host using the :meth:`.GeNNModel.pull_recording_buffers_from_device` method and the spikes emitted by a population 
can be accessed via the :attr:`.NeuronGroupMixin.spike_recording_data` property. Similarly, pre and postsynaptic spike-like events used by a synapse group
can be accessed via the :attr:`.SynapseGroupMixin.pre_spike_event_recording_data` and :attr:`.SynapseGroupMixin.post_spike_event_recording_data` properties, respectively.
For example, the previous example could be extended to record spikes from a :class:`.NeuronGroup` ``pop`` as follows:

..  code-block:: python

    ...
    pop.spike_recording_enabled = True
    
    model.build()
    model.load(num_recording_timesteps=100)

    while model.timestep < 100:
        model.step_time()
    
    model.pull_recording_buffers_from_device()
    spike_times, spike_ids = pop.spike_recording_data[0]

If batching was enabled, spike recording data from batch ``b`` would be accessed with e.g. ``pop.spike_recording_data[b]``.

.. _section-monitoring-activity:

------------------------------
Monitoring simulation activity
------------------------------
When running simulations, you often need to verify that your model is active and progressing correctly.
GeNN provides several mechanisms for monitoring simulation activity:

Timestep and time
-----------------
The current simulation timestep and time can be accessed at any point:

..  code-block:: python

    model.load()

    while model.timestep < 1000:
        model.step_time()

        # Print progress every 100 timesteps
        if model.timestep % 100 == 0:
            print(f"Timestep: {model.timestep}, Time: {model.t} ms")

The :attr:`.GeNNModel.timestep` attribute is an integer counter (starting from 0) that increments
with each call to :meth:`.GeNNModel.step_time`. The :attr:`.GeNNModel.t` attribute gives the
current simulation time in milliseconds, calculated as ``timestep * dt``.

Spike counts
------------
To count the total number of spikes emitted by a population, you can enable spike recording
(see :ref:`section-spike-recording`) and count the recorded spikes:

..  code-block:: python

    pop.spike_recording_enabled = True
    model.load(num_recording_timesteps=100)

    while model.timestep < 100:
        model.step_time()

    model.pull_recording_buffers_from_device()
    spike_times, spike_ids = pop.spike_recording_data[0]

    # Count spikes per neuron
    spike_counts = np.bincount(spike_ids, minlength=pop.size)
    print(f"Total spikes: {len(spike_times)}")
    print(f"Spikes per neuron: {spike_counts}")

Similarly, spike-like events can be counted by enabling :attr:`.NeuronGroup.spike_event_recording_enabled`
and accessing :attr:`.SynapseGroupMixin.pre_spike_event_recording_data` or
:attr:`.SynapseGroupMixin.post_spike_event_recording_data`.

Variable updates
----------------
You can track how state variables evolve over time by pulling them from the device at regular intervals:

..  code-block:: python

    model.load()

    # Store membrane potential over time
    v_history = []

    while model.timestep < 100:
        model.step_time()

        # Sample every 10 timesteps
        if model.timestep % 10 == 0:
            pop.vars["V"].pull_from_device()
            v_history.append(pop.vars["V"].current_values.copy())

    # Convert to numpy array: (timesteps, neurons)
    v_history = np.array(v_history)

For more details on accessing variables, see :ref:`section-pull-push`.

Custom activity counters
------------------------
For more sophisticated activity monitoring, you can add counter variables to your models:

..  code-block:: python

    neuron_model = create_neuron_model(
        "neuron_with_counter",
        vars=[("V", "scalar"), ("spike_count", "unsigned int")],
        sim_code="""
        V += ...;  // Normal dynamics
        """,
        threshold_condition_code="V >= 10.0",
        reset_code="""
        V = 0.0;
        spike_count++;  // Increment counter on each spike
        """)

    pop = model.add_neuron_population("Neurons", 100, neuron_model,
                                      {}, {"V": 0.0, "spike_count": 0})

    # After simulation
    pop.vars["spike_count"].pull_from_device()
    total_spikes = np.sum(pop.vars["spike_count"].current_values)

This approach works for any custom metric: spike counts, event counts, number of updates, etc.

..  note::

    When monitoring simulation activity on GPU backends (CUDA, HIP), note that:

    - Kernels execute asynchronously with the CPU
    - Pulling variables from device triggers synchronization
    - Pulling too frequently can impact performance
    - For production runs, consider sampling at longer intervals

---------
Variables
---------
In real simulations, as well as spikes, you often want to interact with model state variables as the simulation runs.
State variables are encapsulated in :class:`pygenn.model_preprocessor.VariableBase` objects and all populations own dictionaries of these, accessible by variable name.
For example all groups have :attr:`.GroupMixin.vars` whereas, synapse groups additionally have :attr:`.SynapseGroupMixin.pre_vars` and :attr:`.SynapseGroupMixin.post_vars`.
By default, copies of GeNN variables are allocated both on the GPU device and the host from where they can be accessed from Python.
However, if variable's location is set to :attr:`.VarLocation.DEVICE`, they cannot be accessed from Python.

..  _`section-pull-push`:

Pushing and pulling
-------------------
The contents of the host copy of a variable can be 'pushed' to the GPU device by calling :meth:`pygenn.model_preprocessor.ArrayBase.push_to_device`
and 'pulled' from the GPU device into the host copy by calling :meth:`pygenn.model_preprocessor.ArrayBase.pull_from_device`.
In practice this takes the shape of, for example,

..  code-block:: python

    pop.vars["V"].push_to_device()

in order to push the CPU copy of the variable "V" in population ``pop`` to the GPU memory, and

..  code-block:: python
                 
    pop.vars["V"].pull_from_device()

to make the reverse transfer.
    
When using the single-threaded CPU backend, these operations do nothing but we recommend leaving them in place so models will work transparantly across all backends.


Values and views
----------------
To access the data associated with a variable, you can use the ``current_values`` property. For example to save the current values of a variable: 

..  code-block:: python

    np.save("values.npy", pop.vars["V"].current_values)

This will make a copy of the data owned by GeNN and apply any processing required to transform it into a user-friendly format.
For example, state variables associated with sparse matrices will be re-ordered into the same order as the indices used to construct the matrix
and the values from the current delay step will be extracted for per-neuron variables which are accessed from synapse groups with delays.
If you wish to access the values across all delay steps, the ``values`` property can be used.
Additionally, you can can *directly* access the memory owned by GeNN using a 'memory view' for example to set all elements of a variable:

..  code-block:: python

    pop.vars["V"].current_view[:] = 1.0

..  note::

    The memory access is always to the host memory space (unless it is them same as the backend memory space for "single_threaded_cpu" or through zero copy memory). Therefore, typically, memory access would look like

..  code-block:: python

    pop.vars["V"].pull_from_device()
    np.save("values.npy", pop.vars["V"].current_values)

and similarly,

..  code-block:: python

    pop.vars["V"].current_view[:] = 1.0
    pop.vars["V"].push_to_device()

    
.. _section-extra-global-parameters:

-----------------------
Extra global parameters
-----------------------
Extra global parameters behave very much like variables. 
They are encapsulated in :class:`pygenn.model_preprocessor.ExtraGlobalParameter` objects which are derived from the same 
:class:`pygenn.model_preprocessor.ArrayBase` base class and thus share much of the functionality described above.
Populations also own dictionaries of extra global parameters, accessible by name.
For example :class:`.NeuronGroup` has :attr:`.NeuronGroup.extra_global_params` whereas, :class:`.SynapseGroup` has 
:attr:`.SynapseGroup.extra_global_params` to hold extra global parameters associated with the weight update model and
:attr:`.SynapseGroup.psm_extra_global_params` to hold extra global parameters associated with the postsynaptic model.

One very important difference between extra global parameters and variables is that extra global parameters need to be allocated
and provided with initial contents before the model is loaded. For example, to allocate an extra global parameter
called "X" to hold 100 elements which are initially all zero you could do the following:

..  code-block:: python

    ...
    pop.extra_global_params["X"].set_init_values(np.zeros(100))

    model.build()
    model.load()

After allocation, extra global parameters can be accessed just like variables, for example:

..  code-block:: python

    pop.extra_global_params["X"].current_view[:] = 1.0
    pop.extra_global_params["X"].push_to_device()

.. _section-dynamic-parameters:

------------------
Dynamic parameters
------------------
As discussed previously, when building a model, parameters can be made dynamic e.g. by calling :meth:`pygenn.NeuronGroup.set_param_dynamic` on a :class:`.NeuronGroup`.
The values of these parameters can then be set at runtime using the :meth:`pygenn.GroupMixin.set_dynamic_param_value` method. For example to increase the value of a
parameter called "tau" on a population ``pop``, you could do the following:

..  code-block:: python

    ...
    pop.set_param_dynamic("tau")
    
    model.build()
    model.load()
    
    tau = np.arange(0, 100, 10)
    while model.timestep < 100:
        if (model.timestep % 10) == 0:
            pop.set_dynamic_param_value("tau", tau[model.timestep // 10])
            
        model.step_time()
        
        
