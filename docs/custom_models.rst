.. py:currentmodule:: pygenn

=============
Custom models
=============
One of the main things that makes GeNN different from other SNN simulators is that all the 
models and snippets  used to describe the behaviour of your model (see :ref:`section-building-networks`)
can be easily customised by the user using strings containing a C-like language called GeNNCode.

--------
GeNNCode
--------
GeNN model functionality is implemented using strings of C-like code which we call GeNNCode.
This is essentially C99 (https://en.cppreference.com/w/c/language) with the following differences:

- No preprocessor
- Enough support for strings to printf debug messages but not much more i.e. no ``strstr`` etc.
- Functions, typedefines and structures cannot be defined in user code
- Structures are not supported at all
- Some esoteric C99 language features like octal integer and hexadecimal floating point literals aren't supported
- The address of (&) operator isn't supported. On the GPU hardware GeNN targets, local variables are assumed to be stored in registers and not addressable. The only time this is limiting is when dealing with extra global parameter arrays as you can no longer do something like ``const int *egpSubset = &egp[offset];`` and instead have to do ``const int *egpSubset = egp + offset;``.
- Like C++ (but not C99) function overloading is supported so ``sin(30.0f)`` will resolve to the floating point rather than double-precision version.
- Floating point literals like ``30.0`` without a suffix will be treated as ``scalar`` (i.e. the floating point type declared as the precision of the overall model), ``30.0f`` will always be treated as float and ``30.0d`` will always be treated as double.
- An LP64 data model is used on all platforms where ``int`` is 32-bit and ``long`` is 64-bit.
- Only the following standard library functions are supported: ``cos``, ``sin``, ``tan``, ``acos``, ``asin``, ``atan``, ``atan2``, ``cosh``, ``sinh``, ``tanh``, ``acosh``, ``asinh``, ``atanh``, ``exp``, ``expm1``, ``exp2``, ``pow``, ``scalbn``, ``log``, ``log1p``, ``log2``, ``log10``, ``ldexp``, ``ilogb``, ``sqrt``, ``cbrt``, ``hypot``, ``ceil``, ``floor``, ``fmod``, ``round``, ``rint``, ``trunc``, ``nearbyint``, ``nextafter``, ``remainder``, ``fabs``, ``fdim``, ``fmax``, ``fmin``, ``erf``, ``erfc``, ``tgamma``, ``lgamma``, ``copysign``, ``fma``, ``min``, ``max``, ``abs``, ``printf``

Random number generation
------------------------
Random numbers are useful in many forms of custom model, for example as a source of noise or a probabilistic spiking mechanism. 
In GeNN this can be implemented by using the following functions within GeNNCode:

- ``gennrand()`` returns a random 32-bit unsigned integer
- ``gennrand_uniform()`` returns a number drawn uniformly from the interval :math:`[0.0, 1.0]`
- ``gennrand_normal()`` returns a number drawn from a normal distribution with a mean of 0 and a standard deviation of 1.
- ``gennrand_exponential()`` returns a number drawn from an exponential distribution with :math:`\lambda=1`.
- ``gennrand_log_normal(mean, std)`` returns a number drawn from a log-normal distribution with the specified mean and standard deviation.
- ``gennrand_gamma(alpha)`` returns a number drawn from a gamma distribution with the specified shape.
- ``gennrand_binomial(n, p)`` returns a number drawn from a binomial distribution with the specified shape.

-----------------------
Initialisation snippets
-----------------------
Initialisation snippets are GeNNCode to initialise various parts of a GeNN model.
They are configurable by the user with parameters, derived parameters and extra global parameters.
Parameters have a homogeneous numeric value across the population being initialised. 
'Derived parameters' are a mechanism for enhanced efficiency when running neuron models. 
They allow constants used within the GeNNCode implementation of a model to be computed 
from more 'user friendly' parameters provided by the user. For example, a decay to apply 
each timestep could be computed from a time constant provided in a parameter called ``tau`` 
by passing the following keyword arguments to one of the snippet or model creation functions described below:

..  code-block:: python

    params=["tau"],
    derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["tau"]))])


Variable initialisation
-----------------------
New variable initialisation snippets can be defined by calling:

.. autofunction:: pygenn.create_var_init_snippet
    :noindex:

Sparse connectivity initialisation
----------------------------------
Sparse connectivity initialisation snippets can be used to initialise connectivity when using
:attr:`SynapseMatrixType.SPARSE` or :attr:`SynapseMatrixType.BITMASK` connectivity; and to
generate connectivity on the fly when using :attr:`SynapseMatrixType.PROCEDURAL` connectivity.
New sparse connectivity initialisation snippets can be defined by calling:

.. autofunction:: pygenn.create_sparse_connect_init_snippet
    :noindex:

Toeplitz connectivity initialisation
------------------------------------
Toeplitz connectivity initialisation snippets are used to generate convolution-like connectivity 
on the fly when using :attr:`SynapseMatrixType.TOEPLITZ` connectivity.
New Toeplitz connectivity initialisation snippets can be defined by calling:

.. autofunction:: pygenn.create_toeplitz_connect_init_snippet
    :noindex:

------
Models
------
Models extend the snippets described above by adding state. 
They are used to define the behaviour of neurons, synapses and custom updates.

Variable access
---------------
When defining custom models intended to work in batched simulations, it is important to
consider the 'variable access' of state variables which determines if they can contain
different values in each batch or whether the same values are shared between batches.
Because simulations are assumed to run in parallel, if variables are shared they must be
be read-only.
Therefore the following modes are available for variables defined in neuron, weight update,
current source and custom connectivity update models:

.. autoclass:: pygenn.VarAccess
    :noindex:

The situation is further complicated when considering custom update models as not
only do these support operations such as reductions but whether the update itself
is batched or not depends on the types of variables it is attached to via
its variable references. Therefore, so that custom update models can be re-used in
different circumstances, their variables can have the following modes:

.. autoclass:: pygenn.CustomUpdateVarAccess
    :noindex:

.. _section-neuron-models:
Neuron models
-------------
Neuron models define the dynamics and spiking behaviour of populations of neurons.
New neuron models are defined by calling:

.. autofunction:: pygenn.create_neuron_model
    :noindex:

.. _section-weight-update-models:
Weight update models
--------------------
Weight update models define the event-driven and time-driven behaviour of synapses and what output they deliver to postsynaptic (and presynaptic) neurons.
New weight update models are defined by calling:

.. autofunction:: pygenn.create_weight_update_model
    :noindex:

Postsynaptic models
-------------------
The postsynaptic model defines how synaptic input translates into an input current (or other input term for models that are not current based).
They can contain equations defining dynamics that are applied to the (summed) synaptic activation, e.g. an exponential decay over time.
New postsynaptic models are defined by calling:

.. autofunction:: pygenn.create_postsynaptic_model
    :noindex:

Current source models
---------------------
Current source models allow input currents to be injected into neuron models.
New current source models are defined by calling:

.. autofunction:: pygenn.create_current_source_model
    :noindex:

Custom update models
---------------------
Custom update models define operations on model variables that can be triggered on demand by the user.
New custom update models are defined by calling:

.. autofunction:: pygenn.create_custom_update_model
    :noindex:

Custom connectivity update models
---------------------------------
Custom update models define operations on model connectivity that can be triggered on demand by the user.
New custom connectivity update models are defined by calling:

.. autofunction:: pygenn.create_custom_connectivity_update_model
    :noindex:
