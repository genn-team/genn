
[![Build Status](https://gen-ci.inf.sussex.ac.uk/buildStatus/icon?job=GeNN/genn/master)](https://gen-ci.inf.sussex.ac.uk/job/GeNN/genn/master) [![codecov.io](https://codecov.io/github/genn-team/genn/coverage.svg?branch=master)](https://codecov.io/github/genn-team/genn?branch=master) [![DOI](https://zenodo.org/badge/24633934.svg)](https://zenodo.org/badge/latestdoi/24633934)
# GPU-enhanced Neuronal Networks (GeNN)

GeNN is a GPU-enhanced Neuronal Network simulation environment based on code generation for Nvidia CUDA.

## Installation

You can download GeNN either as a zip file of a stable release or a
snapshot of the most recent stable version or the unstable development
version using the Git version control system.

### Downloading a release
Point your browser to https://github.com/genn-team/genn/releases
and download a release from the list by clicking the relevant source
code button. After downloading continue to install GeNN as described in the [GitHub installing section](#installing-genn) below.

### Obtaining a Git snapshot

If it is not yet installed on your system, download and install Git
(http://git-scm.com/). Then clone the GeNN repository from Github
```bash
git clone https://github.com/genn-team/genn.git
```
The github url of GeNN in the command above can be copied from the
HTTPS clone URL displayed on the GeNN Github page (https://github.com/genn-team/genn).

This will clone the entire repository, including all open branches.
By default git will check out the master branch which contains the
source version upon which the next release will be based. There are other 
branches in the repository that are used for specific development 
purposes and are opened and closed without warning.

### Installing GeNN

Installing GeNN comprises a few simple steps [^1] to create the GeNN
development environment:

[^1]: While GeNN models are normally simulated using CUDA on NVIDIA GPUs, if you want to use GeNN on a machine without an NVIDIA GPU, you can skip steps v and vi and use GeNN in "CPU_ONLY" mode.

1.  If you have downloaded a zip file, unpack GeNN.zip in a convenient
    location. Otherwise enter the directory where you downloaded the Git
    repository.

2.  Add GeNN's 'bin' directory to your path, e.g. if you are running Linux or Mac OS X and extracted/downloaded GeNN to
    ``$HOME/GeNN``, this can be done with:
    ```bash
    export PATH=$PATH:$HOME/GeNN/bin
    ```
    to make this change persistent, this can be added to your login script (e.g. `.profile` or `.bashrc`) using your favourite text editor or with:
    ```bash
    echo "export PATH=$PATH:$CUDA_PATH/bin" >> ~/.bash_profile
    ```
     If you are using Windows, the easiest way to modify the path is 
     by using the 'Environment variables' GUI, which can be accessed by clicking start and searching for 
     (by starting to type) 'Edit environment variables for your account'.
     In the upper 'User variables' section, scroll down until you see 'Path',
     select it and click 'Edit'.
     Now add a new directory to the path by clicking 'New' in the 'Edit environment variable' window e.g.:
     ![Screenshot of windows edit environment variable window](/doxygen/images/path_windows.png)
     if GeNN is installed in a sub-directory of your home directory (``%USERPROFILE%`` is an environment variable which points to the current user's home directory) called ``genn``.

3.  Install the C++ compiler on the machine, if not already present.
     For Windows, download Microsoft Visual Studio Community Edition from
     https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx.
     When installing Visual Studio, one should select the 'Desktop 
    development with C++' configuration.
     Mac users should download and set up Xcode from
     https://developer.apple.com/xcode/index.html
     , Linux users should install the GNU compiler collection gcc and g++
     from their Linux distribution repository, or alternatively from
     https://gcc.gnu.org/index.html
     
4.  If your machine has a GPU and you haven't installed CUDA already, 
    obtain a fresh installation of the NVIDIA CUDA toolkit from
    https://developer.nvidia.com/cuda-downloads
    Again, be sure to pick CUDA and C++ compiler versions which are compatible
    with each other. The latest C++ compiler need not necessarily be
    compatible with the latest CUDA toolkit.

5.  GeNN uses the ``CUDA_PATH`` environment variable to determine which 
    version of CUDA to build against. On Windows, this is set automatically when 
    installing CUDA. However, if you choose, you can verify which version is 
    selected by looking for the ``CUDA_PATH`` environment variable in the lower 'System variables' section of the GUI you used to configure the path:
    ![Screenshot of windows edit environment variable window](/doxygen/images/cuda_path_windows.png)
    here, CUDA 10.1 and 11.4 are installed and CUDA 11.4 is selected via ``CUDA_PATH``.
    However, on Linux and Mac you need to set ``CUDA_PATH`` manually with:
    ```bash
    export CUDA_PATH=/usr/local/cuda
    ```
    assuming CUDA is installed in /usr/local/cuda (the standard location 
    on Ubuntu Linux). Again, to make this change persistent, this can
    be added to your login script (e.g. ``.profile`` or ``.bashrc``)

This normally completes the installation. Windows users must close
and reopen their command window so changes to the path take effect.

If you are using GeNN in Windows, the Visual Studio development
environment must be set up within every instance of the CMD.EXE command
window used. One can open an instance of CMD.EXE with the development
environment already set up by navigating to Start - All Programs - 
Microsoft Visual Studio - Visual Studio Tools - x64 Native Tools Command Prompt. You may also wish to
create a shortcut for this tool on the desktop, for convenience.

## Usage

### Sample projects

At the moment, the following C++ example projects are provided with GeNN:

- Self-organisation with STDP in the locust olfactory system \([Nowotny et al. 2005][@Nowotnyetal2005]\):
    - with all-to-all connectivity, using built-in neuron and synapse models \(for benchmarks see [Yavuz et al. 2016][@Yavuzetal2016]\)
    - with sparse connectivity for some synapses, using user-defined neuron-and synapse models \(for benchmarks see [Yavuz et al. 2016][@Yavuzetal2016]\)
    - using BITMASK connectivity
    - using synapses with axonal delays
- Pulse-coupled network of Izhikevich neurons \([Izhikevich 2003][@Izhikevich2003]\) (for benchmarks see [Yavuz et al. 2016][@Yavuzetal2016])

- Genetic algorithm for tracking parameters in a Hodgkin-Huxley model cell

- Classifier based on an abstraction of the insect olfactory system \([Schmuker et al. 2014][@Schmukeretal2014]\)

- Cortical microcircuit model \([Potjans et al. 2014][@Potjans2014]\)

- Toy examples:
    - Single neuron population of Izhikevich neuron(s) receiving Poisson spike trains as input
    - Single neuron population of Izhikevich neuron(s) with no synapses
    - Network of Izhikevich neurons with delayed synapses

In order to get a quick start and run one of the the provided example models, navigate to one of the example project directories in the userproject sub-directory, and then follow the instructions in the README file contained within.

## Simulating a new model

The sample projects listed above are already quite highly integrated examples. If one was to use the library to develop a new C++ model, the following would be done:

1. The neuronal network of interest is defined in a model definition file,
    e.g. ``Example1.cc``.  

2.  Within the the model definition file ``Example1.cc``, the following tasks
    need to be completed:

    1.  The GeNN file ``modelSpec.h`` needs to be included,
        ```c++
        #include "modelSpec.h"
        ```

    2.  The values for initial variables and parameters for neuron and synapse
        populations need to be defined, e.g.
        ```c++
        NeuronModels::PoissonNew::ParamValues poissonParams(
        10.0);      // 0 - firing rate
        ```
        would define the (homogeneous) parameters for a population of Poisson
        neurons [^2].
        [^2]: The number of required parameters and their meaning is defined by the
        neuron or synapse type. Refer to the [User manual](https://genn-team.github.io/genn/documentation/4/html/dc/d05/UserManual.html) for details. We recommend, however, to use comments like
        in the above example to achieve maximal clarity of each parameter's
        meaning.

        If heterogeneous parameter values are required for a particular
        population of neurons (or synapses), they need to be defined as "variables"
        rather than parameters.  See the [User manual](https://genn-team.github.io/genn/documentation/4/html/dc/d05/UserManual.html) for how to define new neuron (or synapse) types and the [Variable initialisation](https://genn-team.github.io/genn/documentation/4/html/d4/dc6/sectVariableInitialisation.html) section for more information on 
        initialising these variables to hetererogenous values.

    3.  The actual network needs to be defined in the form of a function
        ``modelDefinition`` [^3], i.e. 
        ```c++
        void modelDefinition(ModelSpec &model); 
        ```
        [^3]: The name ``modelDefinition`` and its parameter of type ``ModelSpec&``
        are fixed and cannot be changed if GeNN is to recognize it as a
        model definition.

    4.  Inside ``modelDefinition()``, The time step ``DT`` needs to be defined, e.g.
        ```c++
        model.setDT(0.1);
        ```
        \note
        All provided examples and pre-defined model elements in GeNN work with
        units of mV, ms, nF and uS. However, the choice of units is entirely
        left to the user if custom model elements are used.

    [MBody1.cc](userproject/MBody1_project/model/MBody1.cc) shows a typical example of a model definition function. In
    its core it contains calls to ``ModelSpec::addNeuronPopulation`` and
    ``ModelSpec::addSynapsePopulation`` to build up the network. For a full range
    of options for defining a network, refer to the [User manual](https://genn-team.github.io/genn/documentation/4/html/dc/d05/UserManual.html).

3.  The programmer defines their own "simulation" code similar to
    the code in [MBody1Sim.cc](userproject/MBody1_project/model/MBody1Sim.cc). In this code,

    1.  They can manually define the connectivity matrices between neuron groups. 
        Refer to the \ref subsect34 section for the required format of
        connectivity matrices for dense or sparse connectivities.

    2.  They can define input patterns or individual initial values for neuron and 
        / or synapse variables.
        \note
        The initial values or initialisation "snippets" given in the ``modelDefinition`` are automatically applied. 

    3.  They use ``stepTime()`` to run one time step on either the CPU or GPU depending on the options passed to genn-buildmodel.
    
    4.  They use functions like ``copyStateFromDevice()`` etc to transfer the
        results from GPU calculations to the main memory of the host computer
        for further processing.

    5.  They analyze the results. In the most simple case this could just be
        writing the relevant data to output files.

For more details on how to use GeNN, please see [documentation](http://genn-team.github.io/genn/).

If you use GeNN in your work, please cite "Yavuz, E., Turner, J. and Nowotny, T. GeNN: a code generation framework for accelerated brain simulations. Scientific Reports, 6. (2016)"


[@Izhikevich2003]: https://doi.org/10.1109/TNN.2003.820440 "Izhikevich, E. M. Simple model of spiking neurons. IEEE transactions on neural networks 14, 1569–1572 (2003)"

[@Nowotnyetal2005]: https://doi.org/10.1007/s00422-005-0019-7 "Nowotny, T., Huerta, R., Abarbanel, H. D. & Rabinovich, M. I. Self-organization in the olfactory system: one shot odor recognition in insects. Biological cybernetics 93, 436–446 (2005)"

[@Potjans2014]: https://doi.org/10.1093/cercor/bhs358 "Potjans, T. C., & Diesmann, M. The Cell-Type Specific Cortical Microcircuit: Relating Structure and Activity in a Full-Scale Spiking Network Model. Cerebral Cortex, 24(3), 785–806 (2014)"

[@Schmukeretal2014]: https://doi.org/10.1073/pnas.1303053111 "Schmuker, M., Pfeil, T. and Nawrot, M.P. A neuromorphic network for generic multivariate data classification. Proceedings of the National Academy of Sciences, 111(6), pp.2081-2086 (2014)"

[@Yavuzetal2016]: https://doi.org/10.1038%2Fsrep18854 "Yavuz, E., Turner, J. and Nowotny, T. GeNN: a code generation framework for accelerated brain simulations. Scientific reports, 6. (2016)"
