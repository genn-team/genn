Author: Alan Diamond, University of Sussex, 2014

This example project uses GeNN to construct a spiking network for classifying a set of 20 odour classes recorded on a FOX enose

The classifier design is based on an abstraction of the insect olfactory system.

The design is detailed in the paper:

"A GPU-based neuromorphic classifier for chemosensing applications"
 Authors: Alan Diamond, Michael Schmuker, Amalia Berna, Stephen Trowell, Thomas Nowotny

Please cite this publication for any work based on this design.

To cite use of GeNN please use:-
T. Nowotny, “Flexible neuronal network simulation framework using code generation for NVidia® CUDATM,” BMC Neurosci., vol. 12, no. Suppl 1, p. P239, 2011.

If you use the enose data in any way you must cite the data repository where the original data is released (doi: XXXXXXXXXXXXX)

BUILD / RUN INSTRUCTIONS 

Install GeNN from the Github release build, following instruction on setting your PATH etc
You will also need a Python installation of the MDP (statistical and machine learning package) and Matplotlib package. e.g. the Anaconda distribution includes both by default

Start a terminal session

cd to this project directory (userproject/Model_Enose_classifier_project)

To build the spiking network model using the GENN meta compiler type:-

 buildmodel Model_Enose_classifier 0

(change the 0 to 1 for a debug build)

You should only have to do this at the start, or when you change the actual network model  (i.e. editing the file Model_Enose_classifier.cc )

Then, to compile the experiment plus the GeNN created C/CUDA code type:-

make clean && make

(or "make clean debug && make debug" if using debug mode )

Once it compiles you should be able to run the classifier against the included enose recording data.
NB: If you use this data in any way you must cite the data repository where the original data is released (doi: XXXXXXXXXXXXX)

To run the experiment use the "experiment" executable created.
usage: experiment <runOnGPU> <baseDir>

e.g. to run from the current directory on GPU, type

 ./bin/linux/release/experiment 1 .

or if you created a debug build

 ./bin/linux/debug/experiment .


In the file experiment.h there are numerous documented parameters that control settings and how the program behaves

In particular, please note the following :-

1) The Python executable defaults to "python". If you have another, then alter the PYTHON_RUNTIME setting. This exe must be on your system path. Note also that it must be Python 2.x compliant.

2)  the default setup will generate spike raster data in the OUTPUT_DIR once it reaches the test stage (after training complete). It will also display this data as a plot after each presentation. Close the plot window to allow the program to proceed. To alter these defaults use the documented settings 
			FLAG_GENERATE_RASTER_PLOT_DURING_TRAINING_REPEAT, 
			FLAG_GENERATE_RASTER_PLOT_DURING_TESTING, 
			RASTER_FREQ, 
			DISPLAY_RASTER_IMMEDIATELY,  
			RASTER_PLOT_EXTRA

3) The program can run in two modes when it undertakes the test stage of a cross validation. 
The first mode (default) is to use the same static data encoding approach (see the underlying paper) as it used for training.
The second mode will try and classify the continuous timeseries recording. You can switch to this mode by setting
the flag USE_FULL_LENGTH_DELAYED_RECORDINGS_FOR_TEST_STAGE and additionally set RASTER_PLOT_EXTRA to "HEAT" if you are using the raster display function (DISPLAY_RASTER_IMMEDIATELY)


This is how the program strcture works roughly.
The experiment (experiment.cu) controls the experiment at a high level. It mostly does this by instructing the classifier (Model_Enose_classifier.cu) which does the grunt work.

So the experiment first tells the classifier to set up the GPU with the model and synapse data.

Then it chooses the training and test set data.

It runs through the training set , with plasticity ON , telling the classifier to run with the specfied observation and collecting the classifier decision.

Then it runs through the test set with plasticity OFF  and collects the results in various reporting files.

At the highest level it also has a loop where you can cycle through a list of parameter values e.g. some threshold value for the classifier to use. It will then report on the performance for each value. You should be aware that some parameter changes won't actually affect the classifier unless you invoke a re-initialisation of some sort. E.g. anything to do with VRs will require the input data cache to be reset between values, anything to do with non-plastic synapse weights won't get cleared down until you upload a changed set to the GPU etc.

