A neuromorphic network for generic multivariate data classification
===================================================================

This project recreates the spiking classifier proposed in the paper by 
Michael Schmuker, Thomas Pfeil and Martin Paul Nawrota using GeNN. The classifier design is based on an 
abstraction of the insect olfactory system. This example uses the IRIS standard data set 
as a test for the classifier.

To build the model using the GENN meta compiler, navigate to genn/userproject/Model_Schmuker_2014_project and type:

genn-buildmodel.bat Model_Schmuker_2014_classifier.cc

for Windows users (add -d for a debug build), or:

genn-buildmodel.sh Model_Schmuker_2014_classifier.cc

for Linux, Mac and other UNIX users. 

You would only have to do this at the start, or when you change your actual network model, 
i.e. on editing the file, Model_Schmuker_2014_classifier.cc 

Then to compile the experiment and the GeNN created C/CUDA code type:

msbuild Schmuker2014_classifier.vcxproj /p:Configuration=Release

for Windows users (change Release to Debug if using debug mode), or:

make

for Linux, Mac and other UNIX users (add DEBUG=1 if using debug mode).

Once it compiles you should be able to run the classifier against the included Iris dataset by typing:

Schmuker2014_classifier .

for Windows users, or:

./experiment .

for Linux, Mac and other UNIX systems.

This is how it works roughly. The experiment (experiment.cu) controls the experiment at a high level. 
It mostly does this by instructing the classifier (Schmuker2014_classifier.cu) which does the grunt work.

So the experiment first tells the classifier to set up the GPU with the model and the synapse data.

Then it chooses the training and test set data.

It runs through the training set with plasticity ON, telling the classifier to run with the specfied 
observation and collecting the classifier decisions.

Then it runs through the test set with plasticity OFF and collects the results in various reporting files.

At the highest level, it also has a loop where you can cycle through a list of parameter values e.g. some 
threshold value for the classifier to use. It will then report on the performance for each value. 
You should be aware that some parameter changes won't actually affect the classifier unless you invoke 
a re-initialisation of some sort. E.g. anything to do with VRs will require the input data cache to be 
reset between values, anything to do with non-plastic synapse weights won't get cleared down until you 
upload a changed set to the GPU etc.

You should also note there is no option currently to run on CPU, this is not due to the demanding task, but 
just hasn't been tweaked yet to allow for this (a small change).
