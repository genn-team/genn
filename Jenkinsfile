#!groovy​
node {
    // Checkout
    stage("Installation") {
        echo "Checking out GeNN";
        
        // Deleting existing checked out version of GeNN
        sh "rm -rf genn";
        
        dir("genn") {
            // Checkout GeNN into it
            // **NOTE** because we're using multi-branch project URL is substituted here
            checkout scm
        }
        
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
    }
    
    stage("Running tests") {
        // Run automatic tests
        if (isUnix()) {
            echo "${env.PATH}";
            dir("genn/tests") {
                // Run tests
                sh "./run_tests.sh -c";
            }
        } 
    }
    
    stage("Gathering test results") {
        dir("genn/tests") {
            // Process JUnit test output
            junit "**/test_results*.xml";
            
            // Archive compiler output
            archive "msg";
        }
    }
    
    stage("Calculating code coverage") {
        // Calculate coverage
        if (isUnix()) {
            echo "${env.PATH}";
            dir("genn/tests") {
                sh "./calc_coverage.sh -c";
            }
        } 
    }
    
    stage("Uploading coverage summary") {
        dir("genn/tests") {
            sh "curl -s https://codecov.io/bash | bash";
        }
    }
    
}