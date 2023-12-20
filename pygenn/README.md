# A Python interface to GeNN
PyGeNN wraps the C++ GeNN API using SWIG, allowing GeNN to be used either directly from Python or as a backend for higher-level Python APIs such as [PyNN](https://github.com/genn-team/pynn_genn).

### Installing PyGeNN from source on Linux or Mac OSX
 - Either download the latest release of GeNN and extract into your home directory or clone using git from https://github.com/genn-team/genn
 - Navigate to the GeNN directory and build the Python extension with setup tools using ``python setup.py develop`` command. 
 
### Installing PyGeNN from source on Windows
 - Ensure that you have at least Python 3.7.5 and Visual Studio 2019 installed. These instructions assume that the Anaconda platform was used to install Python, but it _should_ be possible to install PyGeNN using suitable versions of Python installed in different way (please let us know if you suceed in doing so!)
 - This process requires a command prompt with the environment correctly configured for both Visual Studio **and** Anaconda. To create one, launch an "x64 Native Tools Command Prompt" from your chosen version of Visual Studio's start menu folder and _activate_ your chosen version of Anaconda by running the ``activate.bat`` in its ``Scripts`` directory. For example, if your user is called "me" and Anaconda is installed in your home directory, you would run ``c:\Users\Me\Anaconda3\Scripts\activate.bat c:\Users\Me\Anaconda3``.
 - Navigate to the GeNN directory and build the Python extension with setup tools using ``python setup.py develop`` command.
