#!groovy​

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
for(b = 0; b < builderNodes.size; b++) {
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
                            
                            // Parse test output for GCC warnings
                            // **NOTE** driving WarningsPublisher from pipeline is entirely undocumented
                            // this is based mostly on examples here https://github.com/kitconcept/jenkins-pipeline-examples
                            // **YUCK** fatal errors aren't detected by the 'GNU Make + GNU C Compiler (gcc)' parser
                            // however JENKINS-18081 fixes this for 
                            // the 'GNU compiler 4 (gcc)' parser at the expense of it not detecting make errors...
                            def parserName = ("mac" in nodeLabel) ? "Apple LLVM Compiler (Clang)" : "GNU compiler 4 (gcc)";
                            step([$class: "WarningsPublisher", 
                                parserConfigurations: [[parserName: parserName, pattern: uniqueMsg]], 
                                unstableTotalAll: '0', usePreviousBuildAsReference: true]); 
                        }
                    } 
                }
                
                buildStep("Gathering test results (" + env.NODE_NAME + ")") {
                    if (isUnix()) {
                        dir("genn/tests") {
                            // Process JUnit test output
                            junit "**/test_results*.xml";
                            
                            // If coverage was emitted
                            def uniqueCoverage = "coverage_" + env.NODE_NAME + ".txt";
                            if(fileExists(uniqueCoverage)) {
                                // Archive it
                                archive uniqueCoverage;
                                
                                // Upload to code cov
                                sh "curl -s https://codecov.io/bash | bash -s - -f " + uniqueCoverage + " -t 04054241-1f5e-4c42-9564-9b99ede08113";
                            }
                        }
                    }
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders
