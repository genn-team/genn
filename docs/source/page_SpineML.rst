.. index:: pair: page; SpineML and SpineCreator
.. _doxid-d2/dba/SpineML:

SpineML and SpineCreator
========================

GeNN now supports simulating models built using `SpineML <http://spineml.github.io/>`__ and includes scripts to fully integrate it with the `SpineCreator <http://spineml.github.io/spinecreator/>`__ graphical editor on Linux, Mac and Windows. After installing GeNN using the instructions in :ref:`Installation <doxid-d8/d99/Installation>`, `build SpineCreator for your platform <http://spineml.github.io/spinecreator/>`__.

From SpineCreator, select Edit->Settings->Simulators and add a new simulator using the following settings (replacing "/home/j/jk/jk421/genn" with the GeNN installation directory on your own system):

.. image:: spinecreator_screenshot.png



.. image:: spinecreator_screenshot.png
	:alt: width=10cm

If you would like SpineCreator to use GeNN in CPU only mode, add an environment variable called "GENN_SPINEML_CPU_ONLY".

The best way to get started using SpineML with GeNN is to experiment with some example models. A number are available `here <https://github.com/SpineML/spineml>`__ although the "Striatal model" uses features not currently supported by GeNN and the two "Brette Benchmark" models use a legacy syntax no longer supported by SpineCreator (or GeNN). Once you have loaded a model, click "Expts" from the menu on the left hand side of SpineCreator, choose the experiment you would like to run and then select your newly created GeNN simulator in the "Setup Simulator" panel:

.. image:: spinecreator_experiment_screenshot.png



.. image:: spinecreator_experiment_screenshot.png
	:alt: width=5cm

Now click "Run experiment" and, after a short time, the results of your GeNN simulation will be available for plotting by clicking the "Graphs" option in the menu on the left hand side of SpineCreator. :ref:`Previous <doxid-d9/d61/Examples>` \| :ref:`Top <doxid-d2/dba/SpineML>` \| :ref:`Next <doxid-d3/d0c/brian2genn>`

