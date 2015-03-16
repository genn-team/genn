Author: Alan Diamond, University of Sussex, 2014

This project recreates using GeNN the spiking classifier design used in the paper

"A neuromorphic network for generic multivariate data classification"
 Authors: Michael Schmuker, Thomas Pfeil, Martin Paul Nawrota
    
The classifier design is based on an abstraction of the insect olfactory system.
This example uses the IRIS stadard data set as a test for the classifier

BUILD / RUN INSTRUCTIONS 

Install GeNN from the internet released build, following instruction on setting your PATH etc

Start a terminal session

cd to this project directory (userproject/Model_Schmuker_2014_project)

To build the model using the GENN meta compiler type:-

 buildmodel.sh Model_Schmuker_2014_classifier 0

(change the 0 to 1 for a debug build)

You should only have to do this at the start, or when you change your actual network model  (i.e. editing the file Model_Schmuker_2014_classifier.cc )

Then to compile the experiment plus the GeNN created C/CUDA code type:-

make clean && make

(or "make clean debug && make debug" if using debug mode )

Once it compiles you should be able to run the classifier against the included Iris dataset.

type

 ./experiment .


This is how it works roughly.
The experiment (experiment.cu) controls the experiment at a high level. It mostly does this by instructing the classifier (Schmuker2014_classifier.cu) which does the grunt work.

So the experiment first tells the classifier to set up the GPU with the model and synapse data.

Then it chooses the training and test set data.

It runs through the training set , with plasticity ON , telling the classifier to run with the specfied observation and collecting the classifier decision.

Then it runs through the test set with plasticity OFF  and collects the results in various reporting files.

At the highest level it also has a loop where you can cycle through a list of parameter values e.g. some threshold value for the classifier to use. It will then report on the performance for each value. You should be aware that some parameter changes won't actually affect the classifier unless you invoke a re-initialisation of some sort. E.g. anything to do with VRs will require the input data cache to be reset between values, anything to do with non-plastic synapse weights won't get cleared down until you upload a changed set to the GPU etc.

You should also note there is no option currently to run on CPU, this is not due to the demanding task, it just hasn't been tweaked yet to allow for this (small change).
