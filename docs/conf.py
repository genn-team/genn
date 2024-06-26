# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from inspect import isfunction, ismethod
import os
import sys

sys.path.insert(0, os.path.abspath("../pygenn"))

# -- Project information -----------------------------------------------------

project = "PyGeNN"
copyright = "2024, James Knight, Thomas Nowotny"
author = "James Knight, Thomas Nowotny"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx_rtd_theme",
    "sphinxarg.ext",
    "nbsphinx"]

napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock imports for readthedocs
autodoc_mock_imports = []

autodoc_typehints = "description"

# Combine __init__ documentation with class to remove need to duplicate in derived
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "sidebarwidth": 450,
}

sphinx_gallery_conf = {
     "examples_dirs": "../userproject",   # path to your example scripts
     "gallery_dirs": "userproject",  # path to where to save gallery generated output
}

# Never actually run tutorial notebooks
nbsphinx_execute = "never"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ["_static"]

primary_domain = "py"

deprecations = {
    ("function", "pygenn.create_neuron_model"): ["param_names", "var_name_types"],
    ("function", "pygenn.create_weight_update_model"): ["param_names", "var_name_types",
                                                        "pre_var_name_types", "post_var_name_types",
                                                        "sim_code", "event_code", "learn_post_code",
                                                        "event_threshold_condition_code"],
    ("function", "pygenn.create_postsynaptic_model"): ["param_names", "var_name_types", 
                                                       "apply_input_code", "decay_code"],
    ("function", "pygenn.create_current_source_model"): ["param_names", "var_name_types"],
    ("function", "pygenn.create_custom_update_model"): ["param_names", "var_name_types"],
    ("function", "pygenn.create_var_init_snippet"): ["param_names"],
    ("function", "pygenn.create_sparse_connect_init_snippet"): ["param_names"],
    ("function", "pygenn.create_toeplitz_connect_init_snippet"): ["param_names"]}

def remove_deprecated_arguments(app, what, name, obj, options, signature, return_annotation):
    if (what, name) in deprecations:
        deprecated_args = deprecations[(what, name)]
        
        # Remove arguments from signature
        for a in deprecated_args:
            signature = signature.replace(f"{a}=None,", "")

        return (signature, return_annotation)

def skip_deprecated_decorator(app, what, name, obj, skip, options):
    # If object is method or function and it has the "is_deprecated" 
    # attribute (put there my our decorator), skip it
    if (ismethod(obj) or isfunction(obj)) and hasattr(obj, "is_deprecated"):
        assert hasattr(obj, "__wrapped__")
        return True
    else:
        return skip

def setup(app):
    app.connect("autodoc-process-signature", remove_deprecated_arguments)
    app.connect("autodoc-skip-member", skip_deprecated_decorator)
