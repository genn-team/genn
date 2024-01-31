.. py:currentmodule:: pygenn

Building networks
=================
The model
---------
A network model is defined as follows:
A :class:`~GeNNModel` must be created with a name and a default precision (see \ref floatPrecision)}:

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

Neuron populations
------------------
Neuron populations are added using the function

model.add_neuron_population(pop_name, num_neurons, neuron, param_space, var_space)

where the arguments are:
`neuron`: The type of neuron model. This should either be a string containing the name of a built in model or user-defined neuron type returned by pygenn.genn_model.create_custom_neuron_class} (see \ref sectNeuronModels).
`pop_name`: Unique name of the neuron population
`num_neurons`: number of neurons in the population
params: Dictionary containing parameters of this neuron type
vars: Dictionary containing initial values or initialisation snippets for variables of this neuron type (see \ref sectVariableInitialisation)

The user may add as many neuron populations as the model necessitates.
They must all have unique names. The possible values for the arguments,
predefined models and their parameters and initial values are detailed
