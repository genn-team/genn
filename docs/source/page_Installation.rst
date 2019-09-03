.. index:: pair: page; Installation
.. _doxid-d8/d99/Installation:

Installation
============

You can download GeNN either as a zip file of a stable release or a snapshot of the most recent stable version or the unstable development version using the Git version control system.



.. _doxid-d8/d99/Installation_1Downloading:

Downloading a release
~~~~~~~~~~~~~~~~~~~~~

Point your browser to `https://github.com/genn-team/genn/releases <https://github.com/genn-team/genn/releases>`__ and download a release from the list by clicking the relevant source code button. Note that GeNN is only distributed in the form of source code due to its code generation design. Binary distributions would not make sense in this framework and are not provided. After downloading continue to install GeNN as described in the :ref:`Installing GeNN <doxid-d8/d99/Installation_1installing>` section below.





.. _doxid-d8/d99/Installation_1GitSnapshot:

Obtaining a Git snapshot
~~~~~~~~~~~~~~~~~~~~~~~~

If it is not yet installed on your system, download and install Git (`http://git-scm.com/ <http://git-scm.com/>`__). Then clone the GeNN repository from Github

.. ref-code-block:: cpp

	git clone https://github.com/genn-team/genn.git

The github url of GeNN in the command above can be copied from the HTTPS clone URL displayed on the GeNN Github page (`https://github.com/genn-team/genn <https://github.com/genn-team/genn>`__).

This will clone the entire repository, including all open branches. By default git will check out the master branch which contains the source version upon which the next release will be based. There are other branches in the repository that are used for specific development purposes and are opened and closed without warning.

As an alternative to using git you can also download the full content of GeNN sources clicking on the "Download ZIP" button on the bottom right of the GeNN Github page (`https://github.com/genn-team/genn <https://github.com/genn-team/genn>`__).





.. _doxid-d8/d99/Installation_1installing:

Installing GeNN
~~~~~~~~~~~~~~~

Installing GeNN comprises a few simple steps to create the GeNN development environment. While GeNN models are normally simulated using CUDA on NVIDIA GPUs, if you want to use GeNN on a machine without an NVIDIA GPU, you can skip steps v and vi and use GeNN in "CPU_ONLY" mode.

(i) If you have downloaded a zip file, unpack GeNN.zip in a convenient location. Otherwise enter the directory where you downloaded the Git repository.

(ii) Add GeNN's "bin" directory to your path, e.g. if you are running Linux or Mac OS X and extracted/downloaded GeNN to $HOME/GeNN, then you can add:

.. ref-code-block:: cpp

	export PATH=$PATH:$HOME/GeNN/bin

to your login script (e.g. ``.profile`` or ``.bashrc``. If you are using WINDOWS, the path should be a windows path as it will be interpreted by the Visual C++ compiler ``cl``, and environment variables are best set using ``SETX`` in a Windows cmd window. To do so, open a Windows cmd window by typing ``cmd`` in the search field of the start menu, followed by the ``enter`` key. In the ``cmd`` window type:

.. ref-code-block:: cpp

	setx PATH "C:\Users\me\GeNN\bin;%PATH%"

where ``C:\Users\me\GeNN`` is the path to your GeNN directory.

(iv) Install the C++ compiler on the machine, if not already present. For Windows, download Microsoft Visual Studio Community Edition from `https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx <https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx>`__. When installing Visual Studio, one should select the 'Desktop development with C++' configuration' and the 'Windows 8.1 SDK' and 'Windows Universal CRT' individual components. Mac users should download and set up Xcode from `https://developer.apple.com/xcode/index.html <https://developer.apple.com/xcode/index.html>`__ Linux users should install the GNU compiler collection gcc and g++ from their Linux distribution repository, or alternatively from `https://gcc.gnu.org/index.html <https://gcc.gnu.org/index.html>`__ Be sure to pick CUDA and C++ compiler versions which are compatible with each other. The latest C++ compiler is not necessarily compatible with the latest CUDA toolkit.

(v) If your machine has a GPU and you haven't installed CUDA already, obtain a fresh installation of the NVIDIA CUDA toolkit from `https://developer.nvidia.com/cuda-downloads <https://developer.nvidia.com/cuda-downloads>`__ Again, be sure to pick CUDA and C++ compiler versions which are compatible with each other. The latest C++ compiler is not necessarily compatible with the latest CUDA toolkit.

(vi) Set the ``CUDA_PATH`` variable if it is not already set by the system, by putting

.. ref-code-block:: cpp

	export CUDA_PATH=/usr/local/cuda

in your login script (or, if CUDA is installed in a non-standard location, the appropriate path to the main CUDA directory). For most people, this will be done by the CUDA install script and the default value of /usr/local/cuda is fine. In Windows, CUDA_PATH is normally already set after installing the CUDA toolkit. If not, set this variable with:

.. ref-code-block:: cpp

	setx CUDA_PATH C:\path\to\cuda

This normally completes the installation. Windows useres must close and reopen their command window to ensure variables set using ``SETX`` are initialised.

If you are using GeNN in Windows, the Visual Studio development environment must be set up within every instance of the CMD.EXE command window used. One can open an instance of CMD.EXE with the development environment already set up by navigating to Start - All Programs - Microsoft Visual Studio - Visual Studio Tools - x64 Native Tools Command Prompt. You may wish to create a shortcut for this tool on the desktop, for convenience.

:ref:`Top <doxid-d8/d99/Installation>` \| :ref:`Next <doxid-d7/d98/Quickstart>`

