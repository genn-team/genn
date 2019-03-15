#!groovy​

import hudson.tasks.test.AbstractTestResultAction

// All the types of build we'll ideally run if suitable nodes exist
def desiredBuilds = [
    ["cuda9", "linux", "x86_64"] as Set,
    ["cuda8", "linux", "x86_64"] as Set,
    ["cuda7", "linux", "x86_64"] as Set, 
    ["cuda6", "linux", "x86_64"] as Set, 
    ["cpu_only", "linux", "x86_64"] as Set,
    ["cuda9", "linux", "x86"] as Set,
    ["cuda8", "linux", "x86"] as Set,
    ["cuda7", "linux", "x86"] as Set, 
    ["cuda6", "linux", "x86"] as Set, 
    ["cpu_only", "linux", "x86"] as Set,
    ["cuda9", "mac"] as Set,
    ["cuda8", "mac"] as Set,
    ["cuda7", "mac"] as Set, 
    ["cuda6", "mac"] as Set, 
    ["cpu_only", "mac"] as Set] 

//--------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------
// Wrapper around setting of GitHUb commit status curtesy of https://groups.google.com/forum/#!topic/jenkinsci-issues/p-UFjxKkXRI
// **NOTE** since that forum post, stage now takes a Closure as the last argument hence slight modification 
void buildStep(String message, Closure closure) {
    stage(message)
    {
        try {
            setBuildStatus(message, "PENDING");
            closure();
        } catch (Exception e) {
            setBuildStatus(message, "FAILURE");
        }
    }
}

void setBuildStatus(String message, String state) {
    // **NOTE** ManuallyEnteredCommitContextSource set to match the value used by bits of Jenkins outside pipeline control
    step([
        $class: "GitHubCommitStatusSetter",
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/genn-team/genn/"],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "continuous-integration/jenkins/branch"],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}

//--------------------------------------------------------------------------
// Entry point
//--------------------------------------------------------------------------
// Build dictionary of available nodes and their labels
def availableNodes = [:]
for(node in jenkins.model.Jenkins.instance.nodes) {
    if(node.getComputer().isOnline() && node.getComputer().countIdle() > 0) {
        availableNodes[node.name] = node.getLabelString().split() as Set
    }
}

// Add master if it has any idle executors
if(jenkins.model.Jenkins.instance.toComputer().countIdle() > 0) {
    availableNodes["master"] = jenkins.model.Jenkins.instance.getLabelString().split() as Set
}

// Loop through the desired builds
def builderNodes = []
for(b in desiredBuilds) {
    // Loop through all available nodes
    for(n in availableNodes) {
        // If, after subtracting this node's labels, all build properties are satisfied
        if((b - n.value).size() == 0) {
            print "${n.key} -> ${b}";
            
            // Add node's name to list of builders and remove it from dictionary of available nodes
            // **YUCK** for some reason tuples aren't serializable so need to add an arraylist
            builderNodes.add([n.key, n.value])
            availableNodes.remove(n.key)
            break
        }
    }
}

//--------------------------------------------------------------------------
// Parallel build step
//--------------------------------------------------------------------------
// **YUCK** need to do a C style loop here - probably due to JENKINS-27421 
def builders = [:]
for(b = 0; b < builderNodes.size(); b++) {
    // **YUCK** meed to bind the label variable before the closure - can't do 'for (label in labels)'
    def nodeName = builderNodes.get(b).get(0)
    def nodeLabel = builderNodes.get(b).get(1)
   
    // Create a map to pass in to the 'parallel' step so we can fire all the builds at once
    builders[nodeName] = {
        node(nodeName) {
            def installationStageName =  "Installation (" + env.NODE_NAME + ")";
            
            // Customise this nodes environment so GeNN and googletest environment variables are set and genn binaries are in path
            // **NOTE** these are NOT set directly using env.PATH as this makes the change across ALL nodes which means you get a randomly mangled path depending on node startup order
            withEnv(["GTEST_DIR=" + pwd() + "/googletest-release-1.8.0/googletest",
                     "PATH+GENN=" + pwd() + "/genn/bin"]) {
                stage(installationStageName) {
                    echo "Checking out GeNN";

                    // Deleting existing checked out version of GeNN
                    sh "rm -rf genn";
                    
                    dir("genn") {
                        // Checkout GeNN into it
                        // **NOTE** because we're using multi-branch project URL is substituted here
                        checkout scm
                    }
                    
                    // **NOTE** only try and set build status AFTER checkout
                    try {
                        setBuildStatus(installationStageName, "PENDING");
                        
                        // If google test doesn't exist
                        if(!fileExists("googletest-release-1.8.0")) {
                            echo "Downloading google test framework";
                            
                            // Download it
                            // **NOTE** wget is not standard on mac
                            //sh "wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz";
                            sh 'curl -OL "https://github.com/google/googletest/archive/release-1.8.0.tar.gz" -o "release-1.8.0.tar.gz"'
                
                            // Unarchive it
                            sh "tar -zxvf release-1.8.0.tar.gz";
                        }
                    } catch (Exception e) {
                        setBuildStatus(installationStageName, "FAILURE");
                    }
                }
                
                buildStep("Running tests (" + env.NODE_NAME + ")") {
                    // Run automatic tests
                    if (isUnix()) {
                        dir("genn/tests") {
                            // **YUCK** if dev_toolset is in node label - add flag to enable newer GCC using dev_toolset (CentOS)
                            def runTestArguments = "";
                            if("dev_toolset" in nodeLabel) {
                                echo "Enabling devtoolset 6 version of GCC";
                                runTestArguments += " -d";
                            }
                            
                            // If node is a CPU_ONLY node add -c option 
                            if("cpu_only" in nodeLabel) {
                                runTestArguments += " -c";
                            }

                            // Run tests
                            def uniqueMsg = "msg_" + env.NODE_NAME;
                            def runTestsCommand = "./run_tests.sh" + runTestArguments + " 1>> \"" + uniqueMsg + "\" 2>> \"" + uniqueMsg + "\"";
                            def runTestsStatus = sh script:runTestsCommand, returnStatus:true;

                            // If tests failed, set failure status
                            if(runTestsStatus != 0) {
                                setBuildStatus("Running tests (" + env.NODE_NAME + ")", "FAILURE");
                            }

                            // Archive output
                            archive uniqueMsg;
                            
                            // Run 'next-generation' warning plugin on results
                            if("mac" in nodeLabel) {
                                recordIssues enabledForFailure: true, tool: clang(pattern: uniqueMsg);
                            }
                            else {
                                recordIssues enabledForFailure: true, tool: gcc4(pattern: uniqueMsg);
                            }
                        }
                    } 
                }
                
                buildStep("Gathering test results (" + env.NODE_NAME + ")") {
                    dir("genn/tests") {
                        // Process JUnit test output
                        junit "**/test_results*.xml";
                        
                        // Get test results from current build
                        AbstractTestResultAction testResultAction = currentBuild.rawBuild.getAction(AbstractTestResultAction.class)
                        if (testResultAction != null) {
                            // If all tests haven't been run, fail build
                            if(testResultAction.totalCount != 59) {
                                setBuildStatus("Gathering test results (" + env.NODE_NAME + ")", "FAILURE");
                            }
                        }
                        else {
                            echo "Test result action doesn't exist";
                        }
                    }
                }
                
                buildStep("Uploading coverage (" + env.NODE_NAME + ")") {
                    dir("genn/tests") {
                        // If coverage was emitted
                        def uniqueCoverage = "coverage_" + env.NODE_NAME + ".txt";
                        if(fileExists(uniqueCoverage)) {
                            // Archive it
                            archive uniqueCoverage;
                            
                            // Upload to code cov
                            sh "curl -s https://codecov.io/bash | bash -s - -n " + env.NODE_NAME + " -f " + uniqueCoverage + " -t 04054241-1f5e-4c42-9564-9b99ede08113";
                        }
                        else {
                            echo uniqueCoverage + " doesn't exist!";
                        }
                    }
                }

                buildStep("Building Python wheels (" + env.NODE_NAME + ")") {
                    dir("genn") {
                        // Build set of dynamic libraries
                        echo "Creating dynamic libraries";
                        def uniqueMakeDynamic = "make_dynamic_" + env.NODE_NAME + ".txt";
                        makeCommand = "make DYNAMIC=1 LIBRARY_DIRECTORY=$PWD/pygenn/genn_wrapper 1>> \"" + uniqueMakeDynamic + "\" 2>> \"" + uniqueMakeDynamic + "\"";
                        def makeStatusCode = sh script:makeCommand, returnStatus:true
                        if(makeStatusCode != 0) {
                            setBuildStatus("Building Python wheels (" + env.NODE_NAME + ")", "FAILURE");
                        }

                        // Archive build message
                        archive uniqueMakeDynamic

                        // If node is a mac, re-label libraries
                        if("mac" in nodeLabel) {
                            sh "for f in pygenn/genn_wrapper/libgenn*.dylib; do install_name_tool -id \"@loader_path/$(basename \$f)\" \$f; done";
                        }

                        // Create virtualenv, install numpy and make Python wheel
                        def uniqueWheel = "wheel_" + env.NODE_NAME + ".txt";
                        echo "Creating Python wheels";
                        script = """
                        virtualenv virtualenv
                        ../virtualenv/bin/activate

                        pip install "numpy>1.6, < 1.15"

                        python setup.py clean --all
                        python setup.py bdist_wheel -D . 1>> "${uniqueWheel}" 2>> "${uniqueWheel}"
                        python setup.py bdist_wheel -D . 1>> "${uniqueWheel}" 2>> "${uniqueWheel}"
                        """
                        def wheelStatusCode = sh script:script, returnStatus:true
                        if(wheelStatusCode != 0) {
                            setBuildStatus("Building Python wheels (" + env.NODE_NAME + ")", "FAILURE");
                        }

                        // Archive wheel message
                        archive uniqueWheel

                        // Archive wheel itself
                        archive "*.whl"
                    }
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders
