#!groovy​

// All the types of build we'll ideally run if suitable nodes exist
def desiredBuilds = [
    ["cuda10", "windows", "python27"] as Set,
    ["cuda9", "windows", "python27"] as Set,
    ["cuda8", "windows", "python27"] as Set,
    ["cuda10", "windows", "python3"] as Set,
    ["cuda9", "windows", "python3"] as Set,
    ["cuda8", "windows", "python3"] as Set,
    ["cuda10", "linux", "x86_64", "python27"] as Set,
    ["cuda9", "linux", "x86_64", "python27"] as Set,
    ["cuda8", "linux", "x86_64", "python27"] as Set,
    ["cpu_only", "linux", "x86_64", "python27"] as Set,
    ["cuda9", "linux", "x86", "python27"] as Set,
    ["cuda8", "linux", "x86", "python27"] as Set,
    ["cpu_only", "linux", "x86", "python27"] as Set,
    ["cuda10", "mac", "python27"] as Set,
    ["cuda9", "mac", "python27"] as Set,
    ["cuda8", "mac", "python27"] as Set,
    ["cpu_only", "mac", "python27"] as Set,
    ["cuda10", "linux", "x86_64", "python3"] as Set,
    ["cuda9", "linux", "x86_64", "python3"] as Set,
    ["cuda8", "linux", "x86_64", "python3"] as Set,
    ["cpu_only", "linux", "x86_64", "python3"] as Set,
    ["cuda9", "linux", "x86", "python3"] as Set,
    ["cuda8", "linux", "x86", "python3"] as Set,
    ["cpu_only", "linux", "x86", "python3"] as Set,
    ["cuda10", "mac", "python3"] as Set,
    ["cuda9", "mac", "python3"] as Set,
    ["cuda8", "mac", "python3"] as Set,
    ["cpu_only", "mac", "python3"] as Set]

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
            withEnv(["GTEST_DIR=" + pwd() + "/googletest-release-1.8.1/googletest",
                     "PATH+GENN=" + pwd() + "/genn/bin"]) {
                stage(installationStageName) {
                    echo "Checking out GeNN";

                    // Deleting existing checked out version of GeNN
                    if(isUnix()) {
                        sh "rm -rf genn";
                    }
                    else {
                        bat script:"rmdir /S /Q genn", returnStatus:true;
                    }

                    dir("genn") {
                        // Checkout GeNN into it
                        // **NOTE** because we're using multi-branch project URL is substituted here
                        // **NOTE** for some reason without extensions, Jenkins leaves branch with detached head (https://stackoverflow.com/questions/44006070/jenkins-gitscm-finishes-the-clone-in-a-detached-head-state-how-can-i-make-sure)
                        checkout([$class: "GitSCM",
                            branches: scm.branches,
                            doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
                            extensions: scm.extensions + [[$class: "LocalBranch", localBranch: "**"]],
                            userRemoteConfigs: scm.userRemoteConfigs
                        ])
                    }
                    
                    // **NOTE** only try and set build status AFTER checkout
                    try {
                        setBuildStatus(installationStageName, "PENDING");
                        
                        // If google test doesn't exist
                        if(!fileExists("googletest-release-1.8.1")) {
                            echo "Downloading google test framework";
                            
                            // Download it
                            httpRequest url:"https://github.com/google/googletest/archive/release-1.8.1.zip", outputFile :"release-1.8.1.zip";
                            
                            // Unarchive it
                            unzip "release-1.8.1.zip";
                        }
                    } catch (Exception e) {
                        setBuildStatus(installationStageName, "FAILURE");
                    }
                }
                
                buildStep("Running tests (" + env.NODE_NAME + ")") {
                    // Run automatic tests
                    def uniqueMsg = "msg_" + env.NODE_NAME + ".txt";
                    dir("genn/tests") {
                        if (isUnix()) {
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
                            // **NOTE** uniqueMsg is in genn directory, NOT tests directory
                            def runTestsCommand = "./run_tests.sh" + runTestArguments + " 1>> \"../" + uniqueMsg + "\" 2>> \"../" + uniqueMsg + "\"";
                            def runTestsStatus = sh script:runTestsCommand, returnStatus:true;

                            // If tests failed, set failure status
                            if(runTestsStatus != 0) {
                                setBuildStatus("Running tests (" + env.NODE_NAME + ")", "FAILURE");
                            }
                            
                        }
                        else {
                            // Run tests
                            // **NOTE** uniqueMsg is in genn directory, NOT tests directory
                            def runTestsCommand = """
                            CALL %VC_VARS_BAT%
                            CALL run_tests.bat >> "..\\${uniqueMsg}" 2>&1;
                            """;
                            def runTestsStatus = bat script:runTestsCommand, returnStatus:true;
                            
                            // If tests failed, set failure status
                            if(runTestsStatus != 0) {
                                setBuildStatus("Running tests (" + env.NODE_NAME + ")", "FAILURE");
                            }
                        }
                    }
                }
                
                buildStep("Gathering test results (" + env.NODE_NAME + ")") {
                    dir("genn/tests") {
                        // Process JUnit test output
                        junit "**/test_results*.xml";
                    }
                }
                
                buildStep("Uploading coverage (" + env.NODE_NAME + ")") {
                    dir("genn/tests") {
                        if(isUnix()) {
                            // If coverage was emitted
                            def uniqueCoverage = "coverage_" + env.NODE_NAME + ".txt";
                            if(fileExists(uniqueCoverage)) {
                                // Upload to code cov
                                sh "curl -s https://codecov.io/bash | bash -s - -n " + env.NODE_NAME + " -f " + uniqueCoverage + " -t 04054241-1f5e-4c42-9564-9b99ede08113";
                            }
                            else {
                                echo uniqueCoverage + " doesn't exist!";
                            }
                        }
                    }
                }

                buildStep("Building Python wheels (" + env.NODE_NAME + ")") {
                    dir("genn") {
                        def uniqueMsg = "msg_" + env.NODE_NAME + ".txt";
                        if(isUnix()) {
                            // Build set of dynamic libraries
                            echo "Creating dynamic libraries";
                            makeCommand = ""
                            if("dev_toolset" in nodeLabel) {
                                makeCommand += ". /opt/rh/devtoolset-6/enable\n"
                            }
                            makeCommand += "make DYNAMIC=1 LIBRARY_DIRECTORY=" + pwd() + "/pygenn/genn_wrapper 1>> \"" + uniqueMsg + "\" 2>> \"" + uniqueMsg + "\"";
                            def makeStatusCode = sh script:makeCommand, returnStatus:true
                            if(makeStatusCode != 0) {
                                setBuildStatus("Building Python wheels (" + env.NODE_NAME + ")", "FAILURE");
                            }

                            // Create virtualenv, install numpy and make Python wheel
                            echo "Creating Python wheels";
                            script = """
                            rm -rf virtualenv
                            virtualenv virtualenv
                            . virtualenv/bin/activate

                            pip install "numpy>1.6, < 1.15"

                            python setup.py clean --all
                            python setup.py bdist_wheel -d . 1>> "${uniqueMsg}" 2>> "${uniqueMsg}"
                            python setup.py bdist_wheel -d . 1>> "${uniqueMsg}" 2>> "${uniqueMsg}"
                            """

                            def wheelStatusCode = sh script:script, returnStatus:true
                            if(wheelStatusCode != 0) {
                                setBuildStatus("Building Python wheels (" + env.NODE_NAME + ")", "FAILURE");
                            }
                        }
                        else {
                            // Build set of dynamic libraries for single-threaded CPU backend
                            echo "Creating dynamic libraries";
                            msbuildCommand = """
                            CALL %VC_VARS_BAT%
                            msbuild genn.sln /p:Configuration=Release_DLL /t:single_threaded_cpu_backend >> "${uniqueMsg}" 2>&1
                            """;
                            
                            // If this isn't a CPU_ONLY node, also build CUDA backend
                            if(!nodeLabel.contains("cpu_only")) {
                                msbuildCommand += "msbuild genn.sln /p:Configuration=Release_DLL /t:cuda_backend >> \"${uniqueMsg}\" 2>&1";
                            }
                            
                            def msbuildStatusCode = bat script:msbuildCommand, returnStatus:true
                            if(msbuildStatusCode != 0) {
                                setBuildStatus("Building Python wheels (" + env.NODE_NAME + ")", "FAILURE");
                            }

                            // Remove existing virtualenv
                            bat script:"rmdir /S /Q virtualenv", returnStatus:true;

                            echo "Creating Python wheels";
                            script = """
                            CALL %VC_VARS_BAT%
                            CALL %ANACONDA_ACTIVATE_BAT%
                            
                            CALL conda install -y swig

                            virtualenv virtualenv
                            pushd virtualenv\\Scripts
                            call activate
                            popd

                            pip install numpy

                            copy /Y lib\\genn*Release_DLL.* pygenn\\genn_wrapper
                            
                            python setup.py clean --all
                            python setup.py bdist_wheel -d . >> "${uniqueMsg}" 2>&1
                            python setup.py bdist_wheel -d . >> "${uniqueMsg}" 2>&1
                            """

                            def wheelStatusCode = bat script:script, returnStatus:true
                            if(wheelStatusCode != 0) {
                                setBuildStatus("Building Python wheels (" + env.NODE_NAME + ")", "FAILURE");
                            }
                        }

                        // Archive wheel itself
                        archive "*.whl"
                    }
                }

                buildStep("Archiving output (" + env.NODE_NAME + ")") {
                    dir("genn") {
                        def uniqueMsg = "msg_" + env.NODE_NAME + ".txt";
                        archive uniqueMsg;

                        // Run 'next-generation' warning plugin on results
                        if("mac" in nodeLabel) {
                            recordIssues enabledForFailure: true, tool: clang(pattern: uniqueMsg);
                        }
                        else if("windows" in nodeLabel){
                            recordIssues enabledForFailure: true, tool: msBuild(pattern: uniqueMsg);
                        }
                        else {
                            recordIssues enabledForFailure: true, tool: gcc4(pattern: uniqueMsg);
                        }

                    }
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders

//--------------------------------------------------------------------------
// Final generation of documentation on master
//--------------------------------------------------------------------------
node("master") {
    buildStep("Building documentation") {
        withEnv(["GENN_PATH=" + pwd() + "/genn"]) {
            dir("genn") {
                // Use credentials for git
                withCredentials([usernamePassword(credentialsId: "genn-jenkins-ci", passwordVariable: "GIT_PASSWORD", usernameVariable: "GIT_USERNAME")]) {
                    // Make documentation, add generated rst files to git and push
                    script = """
                    ./makedoc
                    git add docs/source/*.rst
                    git commit -m "automatic commit of doxyrest documentation"
                    git pull
                    git push --set-upstream https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/genn-team/genn.git ${scm.branches[0]}
                    """

                    def docStatusCode = sh script:script, returnStatus:true
                    if(docStatusCode != 0) {
                        setBuildStatus("Building documentation", "FAILURE");
                    }
                }
            }
        }
    }
}
