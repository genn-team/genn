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
    http://www.visualstudio.com/en-us/downloads. Be sure to select the 'Desktop 
    development with C++' configuration' and the 'Windows 8.1 SDK' and 'Windows 
    Universal CRT' individual components. If your machine has an NVIDIA GPU, 
    then download and install a compatible version of the Nvidia CUDA toolkit 
    from: https://developer.nvidia.com/cuda-downloads. Note that the latest 
    version of Visual Studio is not necessarily compatible with the latest 
    version of the CUDA toolkit.

2. Ensure that the `CUDA_PATH` environment variable is defined and
    points to the location of the Nvidia CUDA toolkit installation; and
    that the CUDA `bin` directory is included in the path. These can
    be checked by using: `ECHO %CUDA_PATH%` and `ECHO %PATH%` respectively 
    (although they are usully set automatically by the CUDA installer on Windows systems). 
    If not, correct this using:
    `SETX CUDA_PATH "[drive]\Program Files\NVIDIA GPU Computing
    Toolkit\CUDA[version]"` and `SETX PATH "%PATH%;%CUDA_PATH%`.

3. Add the `bin` sub-directory of the directory in which GeNN is located to your `PATH` variable. For example,
    if you extracted GeNN to `c:\Users\me\GeNN`, use: `SETX PATH "c:\Users\me\GeNN\bin;%PATH%"`.

4. To access a developer command prompt, use the shortcut link in:
    start menu -\> all programs -\> Microsoft Visual Studio
    -\> Visual Studio Tools -\> x64 Native Tools Command Prompt
    which will launch an instance of cmd.exe with a build environment
    already set up. Alternatively, from any cmd console window, run the
    vscvsrsall.bat script under the Visual C++ directory before
    compiling any projects.

This completes the installation. Note that the command window must be
restarted to initialise the variables set using the `SETX` command.

### LINUX / MAC INSTALL

(1) Unpack GeNN.zip in a convenient location, then download and install
    a compatible version of the Nvidia CUDA toolkit from:
    https://developer.nvidia.com/cuda-downloads and the GNU GCC
    compiler collection and GNU Make build environment if it is not
    already present on the system. Note that the latest versions of GCC
    / Clang / Linux are not necessarily compatible with the latest
    version of the CUDA toolkit.

(2) Ensure that the environment variable `CUDA_PATH` is set to the location of your
    Nvidia CUDA toolkit installation and that the CUDA binary directory is in your path.
    For example, if your CUDA toolkit was installed to `/usr/local/cuda`, you can use: 
    ```
    echo "export CUDA_PATH=/usr/local/cuda" >> ~/.bash_profile 
    echo "export PATH=$PATH:$CUDA_PATH/bin" >> ~/.bash_profile
    ```

(3) Add GeNN's `bin` directory to your $PATH variable. For example, if you extracted GeNN to
    `/home/me/genn`, you can use: 
    `echo "export PATH=$PATH:/home/me/genn/bin" >> ~/.bash_profile`

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
the userproject sub-directory, and then follow the instructions in the
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
    as `MBody1Sim.cc`, etcetera. In
    this code,

-   they define input patterns (e.g. for Poisson neurons like in the
    example)

-   they use `stepTime();` to run one time step on whatever 
    backend the model was built using.

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
