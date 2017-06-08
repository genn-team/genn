[![Build Status](https://gen-ci.inf.sussex.ac.uk/buildStatus/icon?job=GeNN/genn/master)](https://gen-ci.inf.sussex.ac.uk/job/GeNN/genn/master) [![codecov.io](https://codecov.io/github/genn-team/genn/coverage.svg?branch=master)](https://codecov.io/github/genn-team/genn?branch=master)
# GPU-enhanced Neuronal Networks (GeNN)

GeNN is a GPU-enhanced Neuronal Network simulation environment based on
code generation for Nvidia CUDA.

## INSTALLING GeNN 

These instructions are for
installing the release obtained from
https://github.com/genn-team/genn/releases. For full instructions and
cloning git branches of the project, see the documentation available at
http://genn-team.github.io/genn/

### WINDOWS INSTALL

1. Download and unpack GeNN.zip to a convenient location, then download
    and install the Microsoft Visual C++ compiler and IDE from:
    http://www.visualstudio.com/en-us/downloads. Be sure to click custom
    install and select all the Visual C++ tools when installing Visual Studio.
    Then download and install a compatible version of the Nvidia CUDA toolkit
    from: https://developer.nvidia.com/cuda-downloads. Note that the latest
    version of Visual Studio is not necessarily compatible with the latest
    version of the CUDA toolkit.

2. Ensure that the `CUDA_PATH` environment variable is defined, and
    points to the location of the Nvidia CUDA toolkit installation, by
    using: `ECHO %CUDA_PATH%` This variable is usully set during most
    CUDA installations on Windows systems. if not, correct this using:
    `SETX CUDA_PATH "[drive]\Program Files\NVIDIA GPU Computing
    Toolkit\CUDA[version]"`.

3. Define the environment variable `GENN_PATH` to point to the
    directory in which GeNN was located. For example, use: 
    `SETX GENN_PATH "\path\to\genn"`.

4. Add `%GENN_PATH%\lib\bin` to your %PATH% variable. For example,
    use: `SETX PATH "%GENN_PATH%\lib\bin;%PATH%"`.

5. To access a developer command prompt, use the shortcut link in:
    start menu -\> all programs -\> Microsoft Visual Studio
    -\> Visual Studio Tools -\> Native Tools Command Prompt
    which will launch an instance of cmd.exe with a build environment
    already set up. Alternatively, from any cmd console window, run the
    vscvsrsall.bat script under the Visual C++ directory before
    compiling any projects.

This completes the installation. Note that the command window must be
restarted to initialise the variables set using the `SETX` command.

### LINUX / MAC INSTALL

(1) Unpack GeNN.zip in a convenient location, then download and install
    a compatible version of the Nvidia CUDA toolkit from:
    https://developer.nvidia.com/cuda-downloads and install the GNU GCC
    compiler collection and GNU Make build environment if it is not
    already present on the system. Note that the latest versions of GCC
    / Clang / Linux are not necessarily compatible with the latest
    version of the CUDA toolkit.

(2) Set the environment variable `CUDA_PATH` to the location of your
    Nvidia CUDA toolkit installation. For example, if your CUDA toolkit
    was installed to `/usr/local/cuda`, you can use: 
    `echo "export CUDA_PATH=/usr/local/cuda" >> ~/.bash_profile`

(3) Set the environment variable `GENN_PATH` to point to the extracted
    GeNN directory. For example, if you extracted GeNN to
    `/home/me/genn`, then you can use: 
    `echo "export GENN_PATH=/home/me/genn" >> ~/.bash_profile`

(4) Add `$GENN_PATH/lib/bin` to your $PATH variable. For example, you
    can use: 
    `echo "export PATH=$PATH:$GENN_PATH/lib/bin" >> ~/.bash_profile`

This completes the installation.

## USING GeNN 

### SAMPLE PROJECTS

At the moment, the following example projects are provided with GeNN:

-   Self-organisation with STDP in the locust olfactory system \([Nowotny et al. 2005][@Nowotnyetal2005]\):
    -   with all-to-all connectivity, using built-in neuron and synapse
        models \(for benchmarks see [Yavuz et al. 2016][@Yavuzetal2016]\)
    -   with sparse connectivity for some synapses, using user-defined
        neuron-and synapse models \(for benchmarks see [Yavuz et al. 2016][@Yavuzetal2016]\)
    -   using INDIVIDUALID scheme
    -   using delayed synapses
-   Pulse-coupled network of Izhikevich neurons \([Izhikevich 2003][@Izhikevich2003]\)
    (for benchmarks see [Yavuz et al. 2016][@Yavuzetal2016])

-   Genetic algorithm for tracking parameters in a Hodgkin-Huxley model
    cell

-   Classifier based on an abstraction of the insect olfactory system
    \([Schmuker et al. 2014][@Schmukeretal2014]\)

-   Toy examples:
    -   Single neuron population of Izhikevich neuron(s) receiving Poisson
    spike trains as input
    -   Single neuron population of Izhikevich neuron(s) with no
        synapses
    -   Network of Izhikevich neurons with delayed synapses

In order to get a quick start and run one of the the provided example
models, navigate to one of the example project directories in
$GENN\_PATH/userproject/, and then follow the instructions in the
README file contained within.

## SIMULATING A NEW MODEL

The sample projects listed above are already quite highly integrated
examples. If one was to use the library for GPU code generation of their
own model, the following would be done:

a)  The model in question is defined in a file, say `Model1.cc`.

b)  this file needs to

- include `modelSpec.h`
- contains the model's definition in the form of a function 
    `void modelDefinition(NNmodel &model)` 
    (`MBody1.cc`) shows a typical example)

c)  The programmer defines their own modeling code along similar lines
    as `map_classol.*` together with `classol_sim.*`, etcetera. In
    this code,

-   they define the connectivity matrices between neuron groups. (In the
    example here those are read from files).

-   they define input patterns (e.g. for Poisson neurons like in the
    example)

-   they use `stepTimeGPU();` to run one time step on the GPU or
    `stepTimeCPU();` to run one on the CPU. (both versions are
    always compiled). However, mixing the two does not make too much
    sense. The host version uses the same memory whereto results from
    the GPU version are copied (see next point)

-   they use functions like `copyStateFromDevice();` etcetera to obtain
    results from GPU calculations.

-   the simulation code is then produced in the following two steps:
    `genn-buildmodel.[sh|bat] ./modelFile.cc` and `make clean && make`

For more details on how to use GeNN, please see [documentation](http://genn-team.github.io/genn/).

If you use GeNN in your work, please cite 
"Yavuz, E., Turner, J. and Nowotny, T. GeNN: a code generation framework for accelerated brain simulations. Scientific Reports, 6. (2016)"


[@Izhikevich2003]: http://dx.doi.org/10.1109/TNN.2003.820440 "Izhikevich, E. M. Simple model of spiking neurons. IEEE transactions on neural networks 14, 1569–1572 (2003)"

[@Nowotnyetal2005]: http://dx.doi.org/10.1007/s00422-005-0019-7 "Nowotny, T., Huerta, R., Abarbanel, H. D. & Rabinovich, M. I. Self-organization in the olfactory system: one shot odor recognition in insects. Biological cybernetics 93, 436–446 (2005)"

[@Schmukeretal2014]: http://dx.doi.org/10.1073/pnas.1303053111 "Schmuker, M., Pfeil, T. and Nawrot, M.P. A neuromorphic network for generic multivariate data classification. Proceedings of the National Academy of Sciences, 111(6), pp.2081-2086 (2014)"

[@Yavuzetal2016]: http://dx.doi.org/10.1038%2Fsrep18854 "Yavuz, E., Turner, J. and Nowotny, T. GeNN: a code generation framework for accelerated brain simulations. Scientific reports, 6. (2016)"
