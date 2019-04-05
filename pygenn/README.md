# A Python interface to GeNN
PyGeNN wraps the C++ GeNN APU using SWIG, allowing GeNN to be used either directly from Python or as a backend for higher-level Python APIs such as [PyNN](https://github.com/genn-team/pynn_genn). Building PyGeNN is a little involved so we recommend installing a prebuilt binary wheel from

### Installing PyGeNN from binary wheels
 - Select a suitable wheel from the Releases page, for example if you have a Linux system with Python 3.7 and CUDA 10 you would pick ``cuda10-pygenn-0.2-cp37-cp37m-linux_x86_64.whl``. If you do not have CUDA installed, ignore the CUDA version and pick the wheel that matches your platform.
 - Install the wheel using pip ``pip install cuda10-pygenn-0.2-cp37-cp37m-linux_x86_64.whl``
 
### Installing PyGeNN from source on Linux or Mac OSX
 - Either download the latest release of GeNN and extract into your home directory or clone using git from (https://github.com/genn-team/genn)
 - Navigate to the GeNN directory and build GeNN as a dynamic library using `make DYNAMIC=1 LIBRARY_DIRECTORY=\`pwd\`
 - On Mac OS X only, set your newly created library's name with ``install_name_tool -id "@loader_path/libgenn_DYNAMIC.dylib" pygenn/genn_wrapper/libgenn_DYNAMIC.dylib`` (you will need to replace ``libgenn_DYNAMIC`` with ``libgenn_CPU_ONLY_DYNAMIC`` if you do not have an NVIDIA GPU)
 - Build the Python extension with setup tools using ``python setup.py develop`` command
 
### Installing PyGeNN from source on Windows
 - Ensure that you have at least Python 3.5 and Visual Studio 2015 installed (extensions for earlier versions of Python cannot be built using any versions of Visual Studio new enough to support C++11). These instructions assume that the Anaconda platform was used to install Python, but it _should_ be possible using suitable versions of Python installed in different ways..
 - This process requires a command prompt with the environment correctly configured for both Visual Studio and Anaconda. Start an "x64 Native Tools Command Prompt" from the start menu folder for your chosen version of Visual Studio and, _activate_ Anaconda using ``c:\Users\Me\Anaconda3\Scripts\activate.bat c:\Users\Me\Anaconda3`` if your user is called "me" and Anaconda is installed in your home directory.
 - From this command prompt, install SWIG using the ``conda install swig`` command.
 - Build GeNN as a dll using ``msbuild genn.sln /t:Build /p:Configuration=Release_DLL``
 - Copy newly built DLLs into pygenn using ``copy /Y lib\\genn*Release_DLL.* pygenn\\genn_wrapper``
 - Build the Python extension with setup tools using ``python setup.py develop`` command
