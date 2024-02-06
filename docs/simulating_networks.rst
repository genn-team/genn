.. py:currentmodule:: pygenn

===================
Simulating networks
===================
Once you have built a network using the :class:`.GeNNModel` API described in `Building networks`_ and 
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

---------------
Spike recording
---------------
Because recording spikes and spike-like events is a common requirement and their sparse nature can make them inefficient to access,
GeNN has a dedicated events recording system which collects events, emitted over a number of timesteps, in GPU memory before transferring to the host. 
Spike recording can be enabled on chosen neuron groups by setting the :attr:`.NeuronGroup.spike_recording_enabled` and :attr:`.NeuronGroup.spike_event_recording_enabled` properties. 
Memory can then be allocated at runtime for spike recording by using the ``num_recording_timesteps`` keyword argument to :meth:`.GeNNModel.load``.
Spikes can then be copied from the GPU to the host using the :meth:`.GeNNModel.pull_recording_buffers_from_device` method and the spikes emitted by a population 
can be accessed via the :attr:`.NeuronGroup.spike_recording_data` property. Similarly, spike-like events emitted by a population can be accessed via the 
:attr:`.NeuronGroup.spike_event_recording_data` property. For example, the previous example could be extended to record spikes from a :class:`.NeuronGroup` ``pop`` as follows:

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

---------
Variables
---------
In real simulations, as well as spikes, you often want to interact with model state variables as the simulation runs.
Both model state variables and 


-----------------------
Extra global parameters
-----------------------

------------------
Dynamic parameters
------------------

