.. py:currentmodule:: pygenn

=============
Custom models
=============
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

For example, if we wanted to define a snippet to initialise variables by sampling from a normal distribution, 
redrawing if the value is negative (which could be useful to ensure delays remain causal):


..  code-block:: python

    normal_positive_model = pygenn.create_var_init_snippet(
        "normal_positive",
        params=["mean", "sd],
        var_init_code=
            """
            scalar normal;
            do
            {
            normal = mean + (gennrand_normal() * sd);
            } while (normal < 0.0);
            value = normal;
            """)


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

For example, we can define a leaky integrator :math:`\tau\frac{dV}{dt}= -V + I_{{\rm syn}}` solved using Euler's method:

..  code-block:: python

    leaky_integrator_model = pygenn.create_neuron_model(
        "leaky_integrator",

        sim_code=
            """
            V += (-V + Isyn) * (dt / tau);
            """,
        threshold_condition_code="V >= 1.0",
        reset_code=
            """
            V = 0.0;
            """,

        params=["tau"],
        vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE)])


Additional input variables
^^^^^^^^^^^^^^^^^^^^^^^^^^
Normally, neuron models receive the linear sum of the inputs coming from all of their synaptic inputs through the ``Isyn`` variable. 
However neuron models can define additional input variables - allowing input from different synaptic inputs to be combined non-linearly.
For example, if we wanted our leaky integrator to operate on the the product of two input currents, we could modify our model as follows:

..  code-block:: python

    additional_input_vars=[("Isyn2", "scalar", 1.0)],
    sim_code=
        """
        const scalar input = Isyn * Isyn2;
        sim_code="V += (-V + input) * (dt / tau);
        """,


Weight update models
--------------------
Weight update models define the event-driven and time-driven behaviour of synapses and what output they deliver to postsynaptic (and presynaptic) neurons.
New weight update models are defined by calling:

.. autofunction:: pygenn.create_weight_update_model

For example, we can define a simple additive STDP rule with nearest-neighbour spike pairing and the following time-dependence:

..  math::

    \Delta w_{ij} & = \
        \begin{cases}
            A_{+}\exp\left(-\frac{\Delta t}{\tau_{+}}\right) & if\, \Delta t>0\\
            A_{-}\exp\left(\frac{\Delta t}{\tau_{-}}\right) & if\, \Delta t\leq0
        \end{cases}

in a fully event-driven manner as follows:

..  code-block:: python

    stdp_additive_model = pygenn.create_weight_update_model(
        "stdp_additive",
        params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
        vars=[("g", "scalar")],

        pre_spike_syn_code=
            """
            addToPost(g);
            const scalar dt = t - st_post;
            if (dt > 0) {
                const scalar timing = exp(-dt / tauMinus);
                const scalar newWeight = g - (Aminus * timing);
                g = fmax(Wmin, fmin(Wmax, newWeight));
            }
            """,
        post_spike_syn_code=
            """
            const scalar dt = t - st_pre;
            if (dt > 0) {
                const scalar timing = exp(-dt / tauPlus);
                const scalar newWeight = g + (Aplus * timing);
                g = fmax(Wmin, fmin(Wmax, newWeight));
            }
            """)

Pre and postsynaptic dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The memory required for synapse variables and the computational cost of updating them tends to grow with :math:`O(N^2)` with the number of neurons.
Therefore, if it is possible, implementing synapse variables on a per-neuron rather than per-synapse basis is a good idea. 
The ``pre_var_name_types`` and ``post_var_name_types`` keyword arguments} are used to define any pre or postsynaptic state variables.
For example, using pre and postsynaptic variables, our event-driven STDP rule can be extended to use all-to-all spike pairing using pre and postsynaptic _trace_ variables [Morrison2008]_ :

..  code-block:: python

    stdp_additive_2_model = genn_model.create_custom_weight_update_class(
        "stdp_additive_2",
        params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
        vars=[("g", "scalar")],
        pre_vars=[("preTrace", "scalar")],
        post_vars=[("postTrace", "scalar")],
        
        pre_spike_syn_code=
            """
            addToPost(g);
            const scalar dt = t - st_post;
            if(dt > 0) {
                const scalar newWeight = g - (aMinus * postTrace);
                g = fmin(wMax, fmax(wMin, newWeight));
            }
            """,
        post_spike_syn_code=
            """
            const scalar dt = t - st_pre;
            if(dt > 0) {
                const scalar newWeight = g + (aPlus * preTrace);
                g = fmin(wMax, fmax(wMin, newWeight));
            }
            """,

        pre_spike_code=
            """
            preTrace += 1.0;
            """,
        pre_dynamics_code=
            """
            preTrace *= tauPlusDecay;
            """,
        post_spike_code=
            """
            postTrace += 1.0;
            """,
        post_dynamics_code=
            """
            postTrace *= tauMinusDecay;
            """)

Synapse dynamics
^^^^^^^^^^^^^^^^
Unlike the event-driven updates previously described, synapse dynamics code is run for each synapse, each timestep i.e. unlike the others it is time-driven. 
This can be used where synapses have internal variables and dynamics that are described in continuous time, e.g. by ODEs.
However using this mechanism is typically computationally very costly because of the large number of synapses in a typical network. 
By using the ``addToPost()`` and ``addToPostDelay()`` functions discussed in the context of ``pre_spike_syn_code``, the synapse dynamics can also be used to implement continuous synapses for rate-based models.
For example a continous synapse could be added to a weight update model definition as follows:

..  code-block:: python

    synapse_dynamics_code="addToPost(g * V_pre);",

where ``V_pre`` is a presynaptic variable reference.

Spike-like events
^^^^^^^^^^^^^^^^^
As well as time-driven synapse dynamics and spike event-driven updates, GeNN weight update models also support "spike-like events". 
These can be triggered by a threshold condition evaluated on the pre or postsynaptic neuron. 
This typically involves pre or postsynaptic weight update model variables or variable references respectively.

For example, to trigger a presynaptic spike-like event when the presynaptic neuron's voltage is greater than 0.02, the following could be added to a weight update model definition:

..  code-block:: python

    pre_event_threshold_condition_code ="V_pre > -0.02"

\end_toggle_code
Whenever this expression evaluates to true, the event code set using the \add_cpp_python_text{SET_EVENT_CODE() macro,`event_code` keyword argument} is executed. For an example, see WeightUpdateModels::StaticGraded.
Weight update models can indicate whether they require the times of these spike-like-events using the \add_cpp_python_text{SET_NEEDS_PRE_SPIKE_EVENT_TIME() and SET_NEEDS_PREV_PRE_SPIKE_EVENT_TIME() macros, ``is_pre_spike_event_time_required`` and ``is_prev_pre_spike_event_time_required`` keyword arguments}.
These times can then be accessed through the \$(seT_pre) and \$(prev_seT_pre) variables.

Postsynaptic models
-------------------
The postsynaptic models defines how synaptic input translates into an input current (or other input term for models that are not current based).
They can contain equations defining dynamics that are applied to the (summed) synaptic activation, e.g. an exponential decay over time.
New postsynaptic models are defined by calling:

.. autofunction:: pygenn.create_postsynaptic_model

By default, the inputs injected by postsynaptic models are accumulated in ``Isyn`` in the postsynaptic 
neuron but they they can also be directed to additional input variables by setting the :attr:`SynapseGroup.post_target_var` property. 


Current source models
---------------------
pass

Custom update models
---------------------
pass

Custom connectivity update models
---------------------------------

Parallel synapse iteration and removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The main GPU operation that custom connectivity updates expose is the ability to generate per-presynaptic neuron update code. This can be used to implement a very simple model which removes 'diagonals' from the connection matrix:

..  code-block:: python

    remove_diagonal_model = pygenn.create_custom_connectivity_update_model(
        "remove_diagonal",
        row_update_code=
            """
            for_each_synapse {
                if(id_post == id_pre) {
                    remove_synapse();
                    break;
                }
            }
            """)

Parallel synapse creation
^^^^^^^^^^^^^^^^^^^^^^^^^
Similarly you could implement a custom connectivity model which adds diagonals back into the connection matrix like this:

..  code-block:: python

    add_diagonal_model = pygenn.create_custom_connectivity_update_model(
        "add_diagonal",
        row_update_code=
            """
            add_synapse(id_pre);
            """)

One important issue here is that lots of other parts of the model (e.g. other custom connectivity updates or custom weight updates) _might_ have state variables 'attached' to the same connectivity that the custom update is modifying. GeNN will automatically detect this and add and shuffle all these variables around accordingly which is fine for removing synapses but has no way of knowing what value to add synapses with. If you want new synapses to be created with state variables initialised to values other than zero, you need to use variables references to hook them to the custom connectivity update. For example, if you wanted to be able to provide weights for your new synapse, you could update the previous example model like:

..  code-block:: python

    add_diagonal_model = pygenn.create_custom_connectivity_update_model(
        "add_diagonal",
        var_refs=[("g", "scalar")],
        row_update_code=
            """
            add_synapse(id_pre, 1.0);
            """)

Host updates
^^^^^^^^^^^^
Some common connectivity update scenarios involve some computation which can't be easily parallelized. If, for example you wanted to determine which elements on each row you wanted to remove on the host, you can include ``host_update_code`` which gets run before the row update code:

..  code-block:: python

    remove_diagonal_model = pygenn.create_custom_connectivity_update_model(
        "remove_diagonal",
        pre_var_name_types=[("postInd", "unsigned int")],
        row_update_code=
            """
            for_each_synapse {
                if(id_post == postInd) {
                    remove_synapse();
                    break;
                }
            }
            """,
        host_update_code=
            """
            for(unsigned int i = 0; i < num_pre; i++) {
               postInd[i] = i;
            }
            pushpostIndToDevice();
            """)

Within host update code, you have full access to parameters, derived parameters, extra global parameters and pre and postsynaptic variables. By design you do not have access to per-synapse variables or variable references and, currently, you can not access pre and post synaptic variable references as there are issues regarding delays. Each variable has an accompanying push and pull function to copy it to and from the device, for variables this has no parameters as illustrated in the above example and for pointer extra global parameters it has a single parameter specifying the size of the array.

