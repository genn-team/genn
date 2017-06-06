#!groovy​

// All the types of build we'll ideally run if suitable nodes exist
def desiredBuilds = [
    ["cuda8", "linux", "x86_64"] as Set,
    ["cuda7", "linux", "x86_64"] as Set, 
    ["cuda6", "linux", "x86_64"] as Set, 
    ["cpu_only", "linux", "x86_64"] as Set, 
    ["cuda8", "linux", "x86"] as Set,
    ["cuda7", "linux", "x86"] as Set, 
    ["cuda6", "linux", "x86"] as Set, 
    ["cpu_only", "linux", "x86"] as Set,
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

void configureEnvironment() {
    // Setup google test and GeNN environment variables
    env.GTEST_DIR = pwd() + "/googletest-release-1.8.0/googletest";
    env.GENN_PATH = pwd() + "/genn";
    
    // Add GeNN binaries directory to path
    env.PATH += ":" + env.GENN_PATH + "/lib/bin";
    
    echo pwd()
    echo env.GENN_PATH;
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
                // Set environment variables
                configureEnvironment();
                
                // Run automatic tests
                if (isUnix()) {
                    dir("genn/tests") {
                        // Run tests
                        if("cpu_only" in nodeLabel) {
                            sh "./run_tests.sh -c";
                        }
                        else {
                            sh "./run_tests.sh";
                        }
                        
                        // Parse test output for GCC warnings
                        // **NOTE** driving WarningsPublisher from pipeline is entirely undocumented
                        // this is based mostly on examples here https://github.com/kitconcept/jenkins-pipeline-examples
                        // **YUCK** fatal errors aren't detected by the 'GNU Make + GNU C Compiler (gcc)' parser
                        // however JENKINS-18081 fixes this for 
                        // the 'GNU compiler 4 (gcc)' parser at the expense of it not detecting make errors...
                        def parserName = ("mac" in nodeLabel) ? "Apple LLVM Compiler (Clang)" : "GNU compiler 4 (gcc)";
                        step([$class: "WarningsPublisher", 
                            parserConfigurations: [[parserName: parserName, pattern: "msg"]], 
                            unstableTotalAll: '0', usePreviousBuildAsReference: true]); 
                    }
                } 
            }
            
            buildStep("Gathering test results (" + env.NODE_NAME + ")") {
                dir("genn/tests") {
                    // Process JUnit test output
                    junit "**/test_results*.xml";
                    
                    // Archive compiler output
                    archive "msg";
                }
            }
            
            buildStep("Calculating code coverage (" + env.NODE_NAME + ")") {
                // Set environment variables
                configureEnvironment();
                
                // Calculate coverage
                dir("genn/tests") {
                    if (isUnix()) {
                        // Run tests
                        if("cpu_only" in nodeLabel) {
                            sh "./calc_coverage.sh -c";
                        }
                        else {
                            sh "./calc_coverage.sh";
                        }
                    }
                    
                    // Stash coverage txt files so master can combine them all together again
                    stash name: nodeName + "_coverage", includes: "coverage.txt"
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders

//--------------------------------------------------------------------------
// Final combination of results
//--------------------------------------------------------------------------
node {
    buildStep("Uploading coverage summary") {
        // Switch to GeNN test directory so git repo (and hence commit etc) can be detected 
        // and so coverage reports gets deleted with rest of GeNN at install-time
        dir("genn/tests") {
            // Loop through builders
            def lcovCommandLine = "lcov";
            def anyCoverage = false
            for(b = 0; b < builderNodes.size; b++) {
                // **YUCK** meed to bind the label variable before the closure - can't do 'for (label in labels)'
                def nodeName = builderNodes.get(b).get(0)
                def nodeCoverageName = nodeName + "_coverage"
                
                // Create directory
                dir(nodeCoverageName) {
                    // Unstash coverage
                    unstash nodeCoverageName
                    
                    // If coverage file exists in stash
                    if(fileExists("coverage.txt")) {
                        // Add trace file within this directory to command line
                        lcovCommandLine += " --add-tracefile " + nodeCoverageName + "/coverage.txt"
                        anyCoverage = true
                    }
                    else {
                        echo "Coverage file generated by node:" + nodeName + " not found in stash"
                    }
                }
            }
            
            // If any coverage reports were found
            if(anyCoverage) {
                // Finalise lcov command line and execute
                lcovCommandLine += " --output-file combined_coverage.txt"
                sh lcovCommandLine

                // Archive raw coverage report
                archive "combined_coverage.txt"

                // **NOTE** the calc_coverage script massages the gcov output into a more useful form so we want to
                // upload this directly rather than allowing the codecov.io script to generate it's own coverage report
                sh "curl -s https://codecov.io/bash | bash -s - -f combined_coverage.txt -t 04054241-1f5e-4c42-9564-9b99ede08113";
            }
            else {
                echo "No coverage reports found"
            }
        }
    }
}