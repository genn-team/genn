=========
Tutorials
=========

CompNeuro 101
=============
Building spiking neural network models in GeNN

Neurons
-------
Create a model consisting of a population of Izhikevich neurons with heterogeneous parameters, driven by a stimulus current. Simulate and record state variables.

.. toctree::
    :maxdepth: 3
    
    comp_neuro_101/1_neurons.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/comp_neuro_101/1_neurons.ipynb

Synapses
--------
Create a simple balanced random network with two, sparsely connected populations of leaky integrate-and-fire neurons. Simulate and record spikes.

.. toctree::
    :maxdepth: 3
    
    comp_neuro_101/2_synapses.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/comp_neuro_101/2_synapses.ipynb

MNIST inference
===============
Perform MNIST inference by converting a pre-trained ANN to an SNN

Presenting a single image
-------------------------
Create a simple three layer network of integrate-and-fire neurons, densely connected with pre-trained weights. Present a single MNIST image and visualise spiking activity.

.. toctree::
    :maxdepth: 3
    
    mnist_inference/tutorial_1.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mnist_inference/tutorial_1.ipynb

Classifying entire test set
---------------------------
Present entire MNIST test set to previous model and calculate accuracy.

.. toctree::
    :maxdepth: 3
    
    mnist_inference/tutorial_2.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mnist_inference/tutorial_2.ipynb


Improve classification performance
----------------------------------
Use parallel batching and custom updates to improve inference performance by over 30x compared to previous tutorial.

.. toctree::
    :maxdepth: 3
    
    mnist_inference/tutorial_3.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mnist_inference/tutorial_3.ipynb

Insect-inspired MNIST classification
====================================
Train a model of the insect mushroom body using an STDP learning rule to classify MNIST.

Projection Neurons
------------------
Create the first layer of *Projection Neurons* which convert input images into a sparse temporal code.

.. toctree::
    :maxdepth: 3
    
    mushroom_body/1_first_layer.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mushroom_body/1_first_layer.ipynb

Kenyon Cells
------------
Add a second, randomly-connected layer of *Kenyon Cells* to the model.

.. toctree::
    :maxdepth: 3
    
    mushroom_body/2_second_layer.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mushroom_body/2_second_layer.ipynb

Kenyon Cell gain control
------------------------
Add recurrent inhibition circuit, inspired by <i>Giant GABAergic Neuron</i> in locusts, to improve sparse coding of the Kenyon Cells.

.. toctree::
    :maxdepth: 3
    
    mushroom_body/3_second_layer_gain_control.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mushroom_body/3_second_layer_gain_control.ipynb

Mushroom Body Output Neurons
----------------------------
Add *Mushroom Body Output Neurons* with STDP learning and train model on MNIST training set.

.. toctree::
    :maxdepth: 3
    
    mushroom_body/4_third_layer.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mushroom_body/4_third_layer.ipynb

Testing
-------
Create a simplified copy of the model without learning, load in the trained weights and calculate inference accuracy on MNIST test set.

.. toctree::
    :maxdepth: 3
    
    mushroom_body/5_testing.ipynb

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/genn-team/genn/blob/genn_5/docs/tutorials/mushroom_body/5_testing.ipynb
