#!groovy​

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

// **YUCK** for some reason String[].contains() doesn't work in a WEIRD way
Boolean arrayContains(String[] array, String string) {
    for(a in array) {
        if(a == string) {
            return true;
        }
    }
    
    return false;
}

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

print desiredBuilds

// Build dictionary of available nodes and their labels
def availableNodes = [:]
for(node in jenkins.model.Jenkins.instance.nodes) {
    if(node.getComputer().isOnline() && node.getComputer().countIdle() > 0) {
        availableNodes[node.name] = node.getLabelString().split() as Set
    }
}

// Add master
if(node.getComputer().countIdle() > 0) {
    availableNodes["master"] = node.getLabelString().split() as Set
}

print availableNodes

// Loop through the desired builds
def builderNodes = [:]
for(b in desiredBuilds) {
    // Loop through all available nodes
    for(n in availableNodes) {
        // If, after subtracting this node's labels, all build properties are satisfied
        if((b - n.value).size() == 0) {
            print "${n.key} -> ${b}";
            
            // Add node's name to list of builders and remove it from dictionary of available nodes
            builderNodes[n.key] = n.value
            availableNodes.remove(n.key)
            break
        }
    }
}
print builderNodes
error('Stopping early')

def builders = [:]
for (x in labels) {
    // **YUCK** meed to bind the label variable before the closure - can't do 'for (label in labels)'
    def label = x
    
    // Split label into it's constituent parts
    def labelComponents = label.split("\\W*&&\\W*");
    
   
    // Create a map to pass in to the 'parallel' step so we can fire all the builds at once
    builders[label] = {
        node(label=label) {
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
                    def gtestExists = fileExists "googletest-release-1.8.0";
                    if(!gtestExists) {
                        echo "Downloading google test framework";
                        
                        // Download it
                        sh "wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz";
            
                        // Unarchive it
                        sh "tar -zxvf release-1.8.0.tar.gz";
                    }
                    
                    // Setup google test and GeNN environment variables
                    env.GTEST_DIR = pwd() + "/googletest-release-1.8.0/googletest";
                    env.GENN_PATH = pwd() + "/genn";
                    
                    // Add GeNN binaries directory to path
                    env.PATH += ":" + env.GENN_PATH + "/lib/bin";
                } catch (Exception e) {
                    setBuildStatus(installationStageName, "FAILURE");
                }
            }
            
            buildStep("Running tests (" + env.NODE_NAME + ")") {
                // Run automatic tests
                if (isUnix()) {
                    dir("genn/tests") {
                        // Run tests
                        echo "CP:" + ("cpu_only" in labelComponents) ? "YES" : "NO";
                        if(arrayContains(labelComponents, "cpu_only")) {
                            sh "./run_tests.sh -c";
                        }
                        else {
                            sh "./run_tests.sh";
                        }
                        
                        // Parse test output for GCC warnings
                        // **NOTE** driving WarningsPublisher from pipeline is entirely undocumented
                        // this is based mostly on examples here https://github.com/kitconcept/jenkins-pipeline-examples
                        // **YUCK** fatal errors aren't detected by the 'GNU Make + GNU C Compiler (gcc)' parser
                        // however https://issues.jenkins-ci.org/browse/JENKINS-18081 fixes this for 
                        // the 'GNU compiler 4 (gcc)' parser at the expense of it not detecting make errors...
                        step([$class: "WarningsPublisher", 
                            parserConfigurations: [[parserName: "GNU compiler 4 (gcc)", pattern: "msg"]], 
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
            
            buildStep("Calculating code coverage (" + label + ")") {
                // Calculate coverage
                if (isUnix()) {
                    dir("genn/tests") {
                        // Run tests
                        if(arrayContains(labelComponents, "cpu_only")) {
                            sh "./calc_coverage.sh -c";
                        }
                        else {
                            sh "./calc_coverage.sh";
                        }
                    }
                } 
            }
            
            buildStep("Uploading coverage summary (" + env.NODE_NAME + ")") {
                dir("genn/tests") {
                    // **NOTE** the calc_coverage script massages the gcov output into a more useful form so we want to
                    // upload this directly rather than allowing the codecov.io script to generate it's own coverage report
                    sh "bash <(curl -s https://codecov.io/bash) -f coverage.txt -t 04054241-1f5e-4c42-9564-9b99ede08113";
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders
