.. py:currentmodule:: pygenn

=============
Custom models
=============
One of the main things that makes GeNN different than other SNN simulators is that all the 
models and snippets  used to describe the behaviour of your model (see `Building networks`_)
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
- The address of (&) operator isn't supported. On the GPU hardware GeNN targets, local variables are assumed to be stored in registers and not addressable. The only time this is limiting is when dealing with extra global parameter arrays as you can no longer do stuff like ``const int *egpSubset = &egp[offset];`` and instead have to do ``const int *egpSubset = egp + offset;``.
- Like C++ (but not C99) function overloading is supported so ``sin(30.0f)`` will resolve to the floating point rather than double-precision version.
- Floating point literals like ``30.0`` without a suffix will be treated as ``scalar``, ``30.0f`` will always be treated as float and ``30.0d`` will always be treated as double.
- A LP64 data model is used on all platforms where ``int`` is 32-bit and ``long`` is 64-bit.
- Only the following standard library functions are supported: ``cos``, ``sin`, ``tan``, ``acos``, ``asin``, ``atan``, ``atan2``, ``cosh``, ``sinh``, ``tanh``, ``acosh``, ``asinh``, ``atanh``, ``exp``, ``expm1``, ``exp2``, ``pow``, ``scalbn``, ``log``, ``log1p``, ``log2``, ``log10``, ``ldexp``, ``ilogb``, ``sqrt``, ``cbrt``, ``hypot``, ``ceil``, ``floor``, ``fmod``, ``round``, ``rint``, ``trunc``, ``nearbyint``, ``nextafter``, ``remainder``, ``fabs``, ``fdim``, ``fmax``, ``fmin``, ``erf``, ``erfc``, ``tgamma``, ``lgamma``, ``copysign``, ``fma``, ``min``, ``max``, ``abs``,``printf``

Random number generation
------------------------
Random numbers are useful in many forms of custom model. For example as a source of noise or a probabilistic spiking mechanism. 
In GeNN this can be implemented by using the following functions within GeNNCode:

- ``gennrand_uniform()`` returns a number drawn uniformly from the interval :math:`[0.0, 1.0]`
- ``gennrand_normal()`` returns a number drawn from a normal distribution with a mean of 0 and a standard deviation of 1.
- ``gennrand_exponential()`` returns a number drawn from an exponential distribution with :math:`\lambda=1`.
- ``gennrand_log_normal(mean, std)`` returns a number drawn from a log-normal distribution with the specified mean and standard deviation.
- ``gennrand_gamma(alpha)`` returns a number drawn from a gamma distribution with the specified shape.
- ``gennrand_binomial(n, p)`` returns a number drawn from a binomial distribution with the specified shape.


-----
Types
-----
pass

-----------------------
Initialisation snippets
-----------------------
Initialisation snippets are use GeNNCode to initialise various parts of a GeNN model.
They are configurable by the user with parameters, derived parameters and extra global parameters.
Parameters have a homogeneous numeric value across the population being initialised. 
'Derived parameters' are a mechanism for enhanced efficiency when running neuron models. 
They allow constants used within the GeNNCode implementation of a model to be computed 
from more 'user friendly' parameters provided by the user. For example, a decay to apply 
each timestep could be computed from a time constant provided in a parameter called ``tau`` 
by passing the following keyword arguments to one of the snippet or model creation functions described bwlo:

..  code-block:: python

    params=["tau"],
    derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["tau"]))])


Variable initialisation
-----------------------
New variable initialisation snippets can be defined by calling:

.. autofunction:: pygenn.create_var_init_snippet

Sparse connectivity initialisation
----------------------------------
pass

Toeplitz connectivity initialisation
------------------------------------
pass

------
Models
------
Stat


Neuron models
-------------
Neuron models define the dynamics and spiking behaviour of populations of neurons.
New neuron models are defined by calling:

.. autofunction:: pygenn.create_neuron_model

Weight update models
--------------------
Weight update models define the event-driven and time-driven behaviour of synapses and what output they deliver to postsynaptic (and presynaptic) neurons.
New weight update models are defined by calling:

.. autofunction:: pygenn.create_weight_update_model

Postsynaptic models
-------------------
The postsynaptic models defines how synaptic input translates into an input current (or other input term for models that are not current based).
They can contain equations defining dynamics that are applied to the (summed) synaptic activation, e.g. an exponential decay over time.
New postsynaptic models are defined by calling:

.. autofunction:: pygenn.create_postsynaptic_model

Current source models
---------------------
Current source models allow input currents to be injected into neuron models.
New current source models are defined by calling:

.. autofunction:: pygenn.create_current_source_model

Custom update models
---------------------
Custom update models define operations that can
New custom update models are defined by calling:

.. autofunction:: pygenn.create_custom_update_model


Custom connectivity update models
---------------------------------
Custom connectivity updates describe 
New custom connectivity update models are defined by calling:

.. autofunction:: pygenn.create_custom_connectivity_update_model
