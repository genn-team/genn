#!/usr/bin/python

##############################################################
# SpineML to GENN platform independent wrapper               #
# Alex Cope 2017                                             #
#                                                            #
# convert_script_s2g is used to manage passing a SpineML     #
# model to GENN                                              #
##############################################################

# mkdir -p from stack overflow (https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python)
import errno
import os
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

import shutil
import filecmp

# xml parser
import xml.etree.ElementTree as ET

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-w", help="Set the Working Directory")
parser.add_argument("-m", help="Set the Model Directory")
parser.add_argument("-o", help="Set the Output Directory")
parser.add_argument("-e", type=int, default=None, help="Set the Experiment index to run")
parser.add_argument("-p", help="Property change options")
parser.add_argument("-d", help="Delay change options")
parser.add_argument("-c", help="Constant current options")
parser.add_argument("-t", help="Time varying current options")
args = parser.parse_args()

if args.w:
    print("Passed working directory: " + args.w)
else:
    print("Working directory not used")
    #exit(0)

if args.m:
    print("Using model directory: " + args.m)
else:
    print("Model directory required")
    exit(0)

if args.o:
    print("Using output directory: " + args.o)
else:
    print "Output directory required"
    exit(0)

if args.e is not None:
    print("Using experiment index: " + str(args.e))
else:
    print("Experiment index required")
    exit(0)

# check we have a GENN_PATH
genn_path = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
print("GENN_PATH is " + genn_path)

# we need to check that the directories exists and if not create them
#mkdir_p(args.w)
mkdir_p(os.path.join(args.o,"model"))

in_dir = args.m
out_dir = os.path.join(args.o,"model")

# we need to process the model, we have a reference to the experiment, so we can load that and extract the model file, then load that and get the component files:

ns_el = {'sml_el': 'http://www.shef.ac.uk/SpineMLExperimentLayer'}
ns_hnl = {'sml_hnl': 'http://www.shef.ac.uk/SpineMLNetworkLayer'}
ns_lnl = {'sml_lnl': 'http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer'}

# extract model file name from the experiment file
el_tree = ET.parse(os.path.join(in_dir, "experiment" + str(args.e) + ".xml"))
el_root = el_tree.getroot()
model_file_name = el_root.find("sml_el:Experiment",ns_el).find("sml_el:Model",ns_el).get("network_layer_url")
print("Using model file: " + model_file_name)

# extract component file names from model files
nl_tree = ET.parse(os.path.join(in_dir, model_file_name))
nl_root = nl_tree.getroot()
components = []
for pop in nl_root.iterfind("sml_lnl:Population",ns_lnl):
    component_file_name = pop.find("sml_lnl:Neuron",ns_lnl).get("url")
    if not component_file_name == "SpikeSource":
        components.append(component_file_name)
    for proj in pop.iterfind("sml_lnl:Projection",ns_lnl):
        component_file_name = proj.find("sml_lnl:Synapse",ns_lnl).find("sml_lnl:WeightUpdate",ns_lnl).get("url")
        components.append(component_file_name)
        component_file_name = proj.find("sml_lnl:Synapse",ns_lnl).find("sml_lnl:PostSynapse",ns_lnl).get("url")
        components.append(component_file_name)

# remove duplicates by converting to a set and back to a list
components = list(set(components))
for component in components:
    print("Using component file:" + component)

if os.path.isdir(args.m) and os.path.isdir(out_dir):
    for component in components:
        shutil.copy(os.path.join(in_dir,component), out_dir)
    shutil.copy(os.path.join(in_dir,model_file_name), out_dir)
    shutil.copy(os.path.join(in_dir,"experiment" + str(args.e) + ".xml"), out_dir)
    exts = ['bin']
    file_names = [fn for fn in os.listdir(in_dir) if any(fn.endswith(ext) for ext in exts)]
    for file_name in file_names:
        shutil.copy(os.path.join(in_dir,file_name), out_dir)
else:
    print("Model directory does not exist!")
    exit(0)

# check for experiment, model and component changes
recompile = False
if os.path.isdir(os.path.join(out_dir, "prev")):
    # do differences
    if not filecmp.cmp(os.path.join(out_dir, model_file_name),os.path.join(out_dir, "prev", model_file_name)):
        recompile = True
    # if model does not match we may have a different model entirely so stop here
    if recompile == False:
        if not filecmp.cmp(os.path.join(out_dir, "experiment" + str(args.e) + ".xml"),os.path.join(out_dir, "prev", "experiment" + str(args.e) + ".xml")):
            recompile = True
        for component in components:
            if not filecmp.cmp(os.path.join(out_dir, component),os.path.join(out_dir, "prev", component)):
                recompile = True
else:
    recompile = True

if recompile is True:
    print("Recompiling model...")
else:
    print("Model has not changed - no recompile required")

# copy the new version over
if not os.path.isdir(os.path.join(out_dir, "prev")):
    mkdir_p(os.path.join(out_dir, "prev"))

for component in components:
    shutil.copy(os.path.join(out_dir,component), os.path.join(out_dir,"prev"))
shutil.copy(os.path.join(out_dir,model_file_name), os.path.join(out_dir,"prev"))
shutil.copy(os.path.join(out_dir,"experiment" + str(args.e) + ".xml"), os.path.join(out_dir,"prev"))

prog = ""
if os.name == "nt":
    #vcvarsall.bat or vcbuildtools.bat
    # Windows only
    if os.path.isfile("C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\Build\\vcvarsall.bat"):
        prog = '"C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\Build\\vcvarsall.bat" amd64'
    if os.path.isfile("C:\\Program Files (x86)\\Microsoft Visual C++ Build Tools\\vcvarsall.bat"):
        prog = '"C:\\Program Files (x86)\\Microsoft Visual C++ Build Tools\\vcvarsall.bat" amd64'
    if os.path.isfile("C:\\Program Files (x86)\\Microsoft Visual C++ Build Tools\\vcbuildtools.bat"):
        prog = '"C:\\Program Files (x86)\\Microsoft Visual C++ Build Tools\\vcbuildtools.bat" amd64'
    if prog == "":
        print("Windows build config script not found")
        exit(0)
    print("Windows build batch = " + prog)
else:
    prog = "echo NIX"
    print "On Linux / OSX"

# Determine whether we should run GeNN in CPU_ONLY mode
cpu_only = (os.environ.get("GENN_SPINEML_CPU_ONLY") is not None)

# check if GeNN initial compile complete
generate_executable = None
simulate_executable = None
if os.name == "nt":
    config = "Release" if cpu_only else "Release_CUDA"
    generate_executable = "spineml_generator_" + config + ".exe"
    simulate_executable = "spineml_simulator_Release.exe"
    
    backend_target = "single_threaded_cpu_backend" if cpu_only else "cuda_backend"
    genn_library = "genn_Release.lib"
    backend_library = "genn_" + backend_target + "_Release.lib"
    
    if not os.path.isfile(os.path.join(genn_path,"lib",genn_library)):
        print("Compiling LibGeNN")
        os.system(prog + "&& cd " + genn_path + "&&" + "msbuild genn.sln /verbosity:minimal /t:genn /p:Configuration=Release")
    if not os.path.isfile(os.path.join(genn_path,"lib", backend_library)):
        print("Compiling backend")
        os.system(prog + "&& cd " + genn_path + "&&" + "msbuild genn.sln /verbosity:minimal /t:" + backend_target + " /p:Configuration=Release")
    if not os.path.isfile(os.path.join(genn_path,"bin",generate_executable)):
        config = "Release" if cpu_only else "Release_CUDA"
        print("Compiling Generate tool")
        os.system(prog + "&& cd " + genn_path + "&&" + "msbuild spineml.sln /verbosity:minimal /t:spineml_generator /p:Configuration=" + config)
    if not os.path.isfile(os.path.join(genn_path,"bin",simulate_executable)):
        print("Compiling Simulate tool")
        os.system(prog + "&& cd " + genn_path + "&&" + "msbuild spineml.sln /verbosity:minimal /t:spineml_simulator /p:Configuration=Release")
else:
    makefile = "MakefileSingleThreadedCPU" if cpu_only else "MakefileCUDA"
    generate_executable = "spineml_generator_single_threaded_cpu" if cpu_only else "spineml_generator_cuda"
    simulate_executable = "spineml_simulator"

    if not os.path.isfile(os.path.join(genn_path,"bin", generate_executable)):
        print("Compiling Generate tool")
        os.system("cd " + os.path.join(genn_path,"src", "spineml", "generator") + " && make -f " + makefile)
    if not os.path.isfile(os.path.join(genn_path,"bin", simulate_executable)):
        print("Compiling Simulate tool")
        os.system("cd " + os.path.join(genn_path,"src", "spineml", "standalone_simulator") + " && make")

# Recompile if needed
if recompile is True:
    f = open(os.path.join(out_dir,"time.txt"),'w')
    f.write('*Compiling...')
    f.close()
    os.system(prog + "&&" + os.path.join(genn_path,"bin",generate_executable) + " " + os.path.join(out_dir,"experiment" + str(args.e) + ".xml"))

f = open(os.path.join(out_dir,"time.txt"),'w')
f.write('*Running...')
f.close()
os.system(prog + "&&" + os.path.join(genn_path,"bin",simulate_executable) + " " + os.path.join(out_dir,"experiment" + str(args.e) + ".xml"))
