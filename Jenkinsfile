#!groovy​

// only keep 100 builds to prevent disk usage from growing out of control
properties([buildDiscarder(logRotator(artifactDaysToKeepStr: '', 
                           artifactNumToKeepStr: '', 
                           daysToKeepStr: '', 
                           numToKeepStr: '100'))])

// All the types of build we'll ideally run if suitable nodes exist
def desiredBuilds = [
    ["cuda11", "windows"] as Set,
    ["cuda12", "windows"] as Set,
    ["amd", "windows"] as Set,
    ["cuda11", "linux"] as Set,
    ["cuda12", "linux"] as Set,
    ["amd", "linux"] as Set,
    ["cuda11", "mac"] as Set,
    ["cuda12", "mac"] as Set,
    ["amd", "mac"] as Set]

//--------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------
// Wrapper around setting of GitHub commit status curtesy of https://groups.google.com/forum/#!topic/jenkinsci-issues/p-UFjxKkXRI
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
// Build list of available nodes and their labels
def availableNodes = []
for (node in jenkins.model.Jenkins.instance.nodes) {
    if (node.getComputer().isOnline() && node.getComputer().countIdle() > 0) {
        availableNodes.add([node.name, node.getLabelString().split() as Set])
    }
}

// Shuffle nodes so multiple compatible machines get used
Collections.shuffle(availableNodes)

// Loop through the desired builds
def builderNodes = []
for (b in desiredBuilds) {
    // Loop through all available nodes
    for (n = 0; n < availableNodes.size(); n++) {
        // If this node has all desired properties
        if(availableNodes[n][1].containsAll(b)) {
            // Add node's name to list of builders and remove it from dictionary of available nodes
            builderNodes.add(availableNodes[n])
            availableNodes.remove(n)
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
            withEnv(["GTEST_DIR=" + pwd() + "/googletest-release-1.11.0/googletest", "PATH+GENN=" + pwd() + "/genn/bin"]) {
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
                        checkout scm
                    }

                    // **NOTE** only try and set build status AFTER checkout
                    try {
                        setBuildStatus(installationStageName, "PENDING");

                        // If google test doesn't exist
                        if(!fileExists("googletest-release-1.11.0")) {
                            echo "Downloading google test framework";

                            // Download it
                            httpRequest url:"https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip", outputFile :"release-1.11.0.zip";

                            // Unarchive it
                            unzip "release-1.11.0.zip";
                        }
                    } catch (Exception e) {
                        setBuildStatus(installationStageName, "FAILURE");
                    }
                }

                def outputFilename = "${WORKSPACE}/genn/output_${NODE_NAME}.txt";
                def compileOutputFilename = "${WORKSPACE}/genn/compile_${NODE_NAME}.txt";
                def coveragePython = "${WORKSPACE}/genn/coverage_python_${NODE_NAME}.xml";
                def coverageCPP = "${WORKSPACE}/genn/coverage_${NODE_NAME}.txt";
                buildStep("Running unit tests (" + env.NODE_NAME + ")") {
                    // Run automatic tests
                    dir("genn") {
                        if (isUnix()) {
                            // Run tests
                            def runTestsCommand = """
                            rm -f "${outputFilename}"

                            # Clean and build unit tests
                            cd tests/unit
                            make clean all COVERAGE=1 1>> "${outputFilename}" 2>&1

                            # Run tests
                            ./test_coverage --gtest_output="xml:test_results_unit.xml" 1>> "${outputFilename}" 2>&1
                            """;
                            def runTestsStatus = sh script:runTestsCommand, returnStatus:true;

                            // If tests failed, set failure status
                            if(runTestsStatus != 0) {
                                setBuildStatus("Running unit tests (" + env.NODE_NAME + ")", "FAILURE");
                            }

                        }
                        else {
                            // Run tests
                            def runTestsCommand = """
                            CALL %VC_VARS_BAT%
                            DEL "${outputFilename}"

                            msbuild genn.sln /m /t:single_threaded_cpu_backend /verbosity:quiet /p:Configuration=Release

                            msbuild tests/tests.sln /m /verbosity:quiet /p:Configuration=Release

                            PUSHD tests/unit
                            unit_Release.exe --gtest_output="xml:test_results_unit.xml"
                            POPD

                            CALL run_tests.bat >> "${outputFilename}" 2>&1;
                            """;
                            def runTestsStatus = bat script:runTestsCommand, returnStatus:true;

                            // If tests failed, set failure status
                            if(runTestsStatus != 0) {
                                setBuildStatus("Running unit tests (" + env.NODE_NAME + ")", "FAILURE");
                            }
                        }
                    }
                }

                buildStep("Setup virtualenv (${NODE_NAME})") {
                    def cupy = ("cuda11" in nodeLabel) ? "cupy-cuda11x" : "cupy-cuda12x";
                    
                    // Set up new virtualenv
                    echo "Creating virtualenv";
                    sh """
                    rm -rf ${WORKSPACE}/venv
                    ${env.PYTHON} -m venv ${WORKSPACE}/venv
                    . ${WORKSPACE}/venv/bin/activate
                    pip install -U pip
                    pip install numpy scipy pybind11 pytest flaky pytest-cov wheel flake8 bitarray psutil build ${cupy}
                    """;
                }

                buildStep("Installing PyGeNN (${NODE_NAME})") {
                    dir("genn") {
                        // Build PyGeNN module
                        echo "Building and installing PyGeNN";
                        def commandsPyGeNN = """
                        . ${WORKSPACE}/venv/bin/activate
                        python setup.py develop --coverage 2>&1 | tee -a "${compileOutputFilename}" >> "${outputFilename}"
                        """;
                        def statusPyGeNN = sh script:commandsPyGeNN, returnStatus:true;
                        if (statusPyGeNN != 0) {
                            setBuildStatus("Building PyGeNN (${NODE_NAME})", "FAILURE");
                        }
                    }
                }

                
                buildStep("Running feature tests (${NODE_NAME})") {
                    dir("genn/tests/features") {
                        // Run ML GeNN test suite
                        def commandsTest = """
                        . ${WORKSPACE}/venv/bin/activate
                        pytest -s -v --cov ../../pygenn --cov-report=xml:${coveragePython} --junitxml test_results_feature.xml 1>> "${outputFilename}" 2>&1
                        """;
                        def statusTests = sh script:commandsTest, returnStatus:true;
                        if (statusTests != 0) {
                            setBuildStatus("Running tests (${NODE_NAME})", "FAILURE");
                        }
                    }
                }

                buildStep("Gathering test results (${NODE_NAME})") {
                    dir("genn/tests") {
                        // Process JUnit test output
                        junit "**/test_results*.xml";
                    }
                }

                buildStep("Uploading coverage (${NODE_NAME})") {
                    dir("genn/tests") {
                        if(isUnix()) {
                            // Run script to gather together GCOV coverage from unit and feature tests
                            sh './gather_coverage.sh'
                            
                            // Upload to code cov
                            withCredentials([string(credentialsId: "codecov_token_genn", variable: "CODECOV_TOKEN")]) {
                                // Upload Python coverage if it was produced
                                if(fileExists(coveragePython)) {
                                    sh 'curl -s https://codecov.io/bash | bash -s - -n ' + env.NODE_NAME + ' -f ' + coveragePython + ' -t $CODECOV_TOKEN';
                                }
                                else {
                                    echo coveragePython + " doesn't exist!";
                                }
                                
                                // Upload CPP coverage if it was produced
                                if(fileExists(coverageCPP)) {
                                    sh 'curl -s https://codecov.io/bash | bash -s - -n ' + env.NODE_NAME + ' -f ' + coverageCPP + ' -t $CODECOV_TOKEN';
                                }
                                else {
                                    echo coverageCPP + " doesn't exist!";
                                }
                            }
                        }
                    }
                }

                buildStep("Building Python wheels (${NODE_NAME})") {
                    dir("genn") {
                        script = """
                            . ${WORKSPACE}/venv/bin/activate

                            python -m build  --wheel . 1>> "${outputFilename}" 2>&1
                            """
                        def wheelStatusCode = sh script:"python -m build --wheel" returnStatus:true
                        if(wheelStatusCode != 0) {
                            setBuildStatus("Building Python wheels (" + env.NODE_NAME + ")", "FAILURE");
                        }

                        // Archive wheel itself
                        archive "*.whl"
                    }
                }

                buildStep("Archiving output (${NODE_NAME})") {
                    dir("genn") {
                        def outputPattern = "output_" + env.NODE_NAME + ".txt";
                        archive outputPattern;

                        // Run 'next-generation' warning plugin on results
                        def compilePattern = "compile_" + env.NODE_NAME + ".txt";
                        if("mac" in nodeLabel) {
                            recordIssues enabledForFailure: true, tool: clang(pattern: compilePattern);
                        }
                        else if("windows" in nodeLabel){
                            recordIssues enabledForFailure: true, tool: msBuild(pattern: compilePattern);
                        }
                        else {
                            recordIssues enabledForFailure: true, tool: gcc4(pattern: compilePattern);
                        }

                    }
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders
