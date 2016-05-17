# GPU enhanced Neuronal Network (GeNN)

GeNN is a GPU enhanced Neuronal Network simulation environment based on
code generation for Nvidia CUDA.

## INSTALLING GeNN 

These instructions are for
installing the release obtained from
https://github.com/genn-team/genn/releases. For full instructions and
cloning git branches of the project, see
http://genn-team.github.io/genn/documentation/html/d8/d99/Installation.html

###WINDOWS INSTALL

1. Download and unpack GeNN.zip to a convenient location, then download
    and install the Microsoft Visual C++ compiler and IDE from:
    http://www.visualstudio.com/en-us/downloads then download and
    install the Nvidia CUDA toolkit from:
    https://developer.nvidia.com/cuda-downloads

2. Ensure that the "CUDA\_PATH" environment variable is defined, and
    points to the location of the Nvidia CUDA toolkit installation, by
    using: `ECHO %CUDA\_PATH%` 
    This variable is usully set during most
    CUDA installations on Windows systems. if not, correct this using:
    `SETX CUDA\_PATH "[drive]\Program Files\NVIDIA GPU Computing
    Toolkit\CUDA[version]"`

3. Define the environment variable "GENN\_PATH" to point to the
    directory in which GeNN was located. For example, use: 
    `SETX GENN\_PATH "\path\to\genn"`

4. Add "%GENN\_PATH%\lib\bin" to your %PATH% variable. For example,
    use: 
    `SETX PATH "%GENN\_PATH%\lib\bin;%PATH%"`

5. Define VC\_PATH as the path to your most recent Visual Studio
    installation, e.g. `setx VC\_PATH "C:\Program Files
    (x86)\Microsoft Visual Studio 10.0"`

Alternatively you can do one of the following:

i.  Run the vscvsrsall.bat script under Visual C++ directory before
    projects are compiled and run in a given cmd.exe terminal window.

ii. Alternatively, one can use the shortcut link in: start menu -\> all
    programs -\> Microsoft Visual Studio -\> Visual Studio Tools -\>
    Visual Studio Command Prompt which will launch an instance of
    cmd.exe in which the vcvarsall.bat compiler setup script has already
    been executed.

This completes the installation.

###LINUX / MAC INSTALL

(1) Unpack GeNN.zip in a convenient location, then download and install
    the Nvidia CUDA toolkit from:
    https://developer.nvidia.com/cuda-downloads and install the GNU GCC
    compiler collection and GNU Make build environment if it is not
    already present on the system.

(2) Set the environment variable "CUDA\_PATH" to the location of your
    Nvidia CUDA toolkit installation. For example, if your CUDA toolkit
    was installed to "/usr/local/cuda", you can use: 
    `echo "export CUDA\_PATH=/usr/local/cuda" \>\> \~/.bash\_profile`

(3) Set the environment variable "GENN\_PATH" to point to the extracted
    GeNN directory. For example, if you extracted GeNN to
    "/home/me/genn", then you can use: 
    `echo "export GENN\_PATH=/home/me/genn" \>\> \~/.bash\_profile`

(4) Add "$GENN_PATH/lib/bin" to your $PATH variable. For example, you
    can use: 
    `echo "export PATH=$PATH:$GENN\_PATH/lib/bin" \>\> \~/.bash\_profile`

This completes the installation.

## USING GeNN 

###SAMPLE PROJECTS

At the moment, the following example projects are provided with GeNN:

-   Locust olfactory system example \([Nowotny et al. 2005][@Nowotnyetal2005]\):
    -   with all-to-all connectivity, using built-in neuron and synapse
        models \(benchmarked in [Yavuz et al. 2016][@Yavuzetal2016]\)
    -   with sparse connectivity for some synapses, using user-defined
        neuron-and synapse models \(benchmarked in [Yavuz et al. 2016][@Yavuzetal2016]\)
    -   using INDIVIDUALID scheme
    -   using delayed synapses
-   Single neuron population of Izhikevich neuron(s) receiving Poisson
    spike trains as input
-   Pulse-coupled network of Izhikevich neurons \([Izhikevich 2003][@Izhikevich2003]\)
    (benchmarked in [Yavuz et al. 2016][@Yavuzetal2016])
    -   with fixed delays (original model)\
    -   Izhikevich with delayed synapses
-   Genetic algorithm for tracking parameters in a Hodgkin-Huxley model
    cell

-   Classifier based on an abstraction of the insect olfactory system
    \([Schmuker et al. 2014][@Schmukeretal2014]\)

-   Toy examples:
    -   Izhikevich network receiving Poisson input spike trains
    -   Single neuron population of Izhikevich neuron(s) with no
        synapses
    -   Network of Izhikevich neurons with delayed synapses

In order to get a quick start and run one of the the provided example
models, navigate to one of the example project directories in
\$GENN\_PATH/userproject/, and then follow the instructions in the
README file contained within.

## SIMULATING A NEW MODEL

The sample projects listed above are already quite highly integrated
examples. If one was to use the library for GPU code generation of their
own model, the following would be done:

a)  The model in question is defined in a file, say "Model1.cc".

b)  this file needs to

- define `DT`
- include `modelSpec.h` and `modelSpec.cc`
- contains the model's definition in the form of a function 
    `void modelDefinition(NNmodel &model)` 
    ("MBody1.cc") shows a typical example)

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

For more details on how to use GeNN, please see the [documentation](http://genn-team.github.io/genn/documentation/html/index.html).

If you use GeNN in your work, please cite 
"Yavuz, E., Turner, J. and Nowotny, T. GeNN: a code generation framework for accelerated brain simulations. Scientific Reports, 6. (2016)"


[@Izhikevich2003]: http://dx.doi.org/10.1109/TNN.2003.820440 "Izhikevich, E. M. Simple model of spiking neurons. IEEE transactions on neural networks 14, 1569–1572 (2003)"

[@Nowotnyetal2005]: http://dx.doi.org/10.1007/s00422-005-0019-7 "Nowotny, T., Huerta, R., Abarbanel, H. D. & Rabinovich, M. I. Self-organization in the olfactory system: one shot odor recognition in insects. Biological cybernetics 93, 436–446 (2005)"

[@Schmukeretal2014]: http://dx.doi.org/10.1073/pnas.1303053111 "Schmuker, M., Pfeil, T. and Nawrot, M.P. A neuromorphic network for generic multivariate data classification. Proceedings of the National Academy of Sciences, 111(6), pp.2081-2086 (2014)"

[@Yavuzetal2016]: http://dx.doi.org/10.1038%2Fsrep18854 "Yavuz, E., Turner, J. and Nowotny, T. GeNN: a code generation framework for accelerated brain simulations. Scientific reports, 6. (2016)"

