#!/bin/bash
#call this as:
#$ bash testprojects.sh "what is new in this run" 2>&1|tee -a outputtestscript
#then:
#$ grep -i warning outputtestscript
#$ grep -i error outputtestscript
#$ grep -i seg outputtestscript
#
custommsg=$1 #Reminder about what is being tried, to be written in the .time file
set -e #exit if error or segfault
BmDir=$GENN_PATH/userproject/refProjFiles

printf "This script may take a while to finish all tests if you use it with default parameters. \n\
You may consider shortening the duration of simulation in classol_sim.h for the MBody examples.\n\n"
firstrun=false;
reuse=1;

echo "Making tools..."
cd tools
make clean && make 
cd ..

echo "firstrun " $firstrun

printf "\nTEST 1: TEST BY CREATING NEW INPUT PATTERNS"
printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing MBody1 ****************************************\n"
printf "\n\n*******************************************************************************\n"
echo "Checking if benchmarking directory exists..."
if [ -d "$BmDir/MBody1" ]; then
  echo "benchmarking directory exists. Using input data from" $BmDir/MBody1 
  printf "\n"
  firstrun_MB1=false;
else
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/MBody1 " and running the test only once (first run for reference)."
  mkdir -p $BmDir/MBody1
  printf "\n"
  firstrun_MB1=true;
fi 
cd MBody1_project
make clean && make 
printf "\n\n####################### MBody1 GPU ######################\n"
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
./generate_run 1 100 1000 20 100 0.0025 testing MBody1 REUSE=$reuse
cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### MBody1 CPU ######################\n"
./generate_run 0 100 1000 20 100 0.0025 testing MBody1 REUSE=$reuse
cp testing_output/testing.out.st testing_output/testing.out.st.CPU 

if [ "$firstrun_MB1" = true ]; then
  printf "Benchmarking is run for the first time. Creating reference input files with the results of these runs. \nIf any error occurs please delete the benchmarking directory and run the script after the values are corrected.\nCopying reference files...\n"
  cp -R ../MBody1_project/testing_output $BmDir/MBody1
 #other MB variants
else
  cp -R testing_output/testing.time $BmDir/MBody1/testing.time
fi

printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing MBody_individualID *********************************\n"
printf "\n\n*******************************************************************************\n"

echo "Checking if benchmarking directory exists..."
if [ -d "$BmDir/MBody_individualID" ]; then
  echo "benchmarking directory exists. Using input data from" $BmDir/MBody_individualID 
  printf "\n"
  firstrun_MBI=false;
else
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/MBody_individualID " and running the test only once (first run for reference)."
  mkdir -p $BmDir/MBody_individualID
  printf "\n"
  firstrun_MBI=true;
fi 

cd ../MBody_individualID_project
make clean && make
printf "\n\n####################### MBody_individualID GPU ######################\n"
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
./generate_run 1 100 1000 20 100 0.0025 testing MBody_individualID REUSE=$reuse
 
cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### MBody_individualID CPU ######################\n"
./generate_run 0 100 1000 20 100 0.0025 testing MBody_individualID REUSE=$reuse
 
cp testing_output/testing.out.st testing_output/testing.out.st.CPU 

if [ "$firstrun_MBI" = true ]; then
  printf "Benchmarking is run for the first time. Creating reference input files with the results of these runs. \nIf any error occurs please delete the benchmarking directory and run the script after the values are corrected.\nCopying reference files...\n"
  cp -R ../MBody_individualID_project/testing_output $BmDir/MBody_individualID
 #other MB variants
else
  cp -R testing_output/testing.time $BmDir/MBody_individualID/testing.time
fi

#MBody1 reference files will be used in the other variants of the MBody project.
printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing MBody_userdef *********************************\n"
printf "\n\n*******************************************************************************\n"
if [ ! -d "$BmDir/MBody_userdef" ]; then
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/MBody_userdef 
  mkdir -p $BmDir/MBody_userdef
  printf "\n"
fi 

cd ../MBody_userdef_project
make clean && make
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### MBody_userdef GPU ######################\n"
./generate_run 1 100 1000 20 100 0.0025 testing MBody_userdef REUSE=$reuse

cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### MBody_userdef CPU ######################\n"
./generate_run 0 100 1000 20 100 0.0025 testing MBody_userdef REUSE=$reuse

cp testing_output/testing.out.st testing_output/testing.out.st.CPU

cp -R testing_output/testing.time $BmDir/MBody_userdef/testing.time

printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing MBody_delayedSyn *********************************\n"
printf "\n\n*******************************************************************************\n"
if [ ! -d "$BmDir/MBody_delayedSyn" ]; then
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/MBody_delayedSyn 
  mkdir -p $BmDir/MBody_delayedSyn
  printf "\n"
fi 

cd ../MBody_delayedSyn_project
make clean && make
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### MBody_delayedSyn GPU ######################\n"
./generate_run 1 100 1000 20 100 0.0025 testing MBody_delayedSyn REUSE=$reuse
 
cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### MBody_delayedSyn CPU ######################\n"
./generate_run 0 100 1000 20 100 0.0025 testing MBody_delayedSyn REUSE=$reuse

cp testing_output/testing.out.st testing_output/testing.out.st.CPU

cp -R testing_output/testing.time $BmDir/MBody_delayedSyn/testing.time

printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing Izh_sparse 10K neurons****************************\n"
printf "\n\n*******************************************************************************\n"
if [ -d "$BmDir/Izh_sparse" ]; then
  echo "benchmarking directory exists at " $BmDir/Izh_sparse 
  printf "\n"
  firstrun_IZH=false;
else
  mkdir -p $BmDir/Izh_sparse
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/Izh_sparse " and running the test only once (first run for reference)."
  printf "\n"
  firstrun_IZH=true;
fi 

cd ../Izh_sparse_project
make clean && make
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### Izh_sparse 10K GPU ######################\n"
./generate_run 1 10000 1000 1 testing Izh_sparse 1.0 REUSE=$reuse

#cp testing_output/testing.out.st testing_output/testing.out.st.GPU
printf "\n\n####################### Izh_sparse 10K CPU ######################\n"
./generate_run 0 10000 1000 1 testing Izh_sparse 1.0 REUSE=$reuse

#cp testing_output/testing.out.st testing_output/testing.out.st.CPU
if [ "$firstrun_IZH" = true ]; then
  cp -R inputfiles inputfiles10K
fi  
if [ "$firstrun_IZH" = true ]; then
  printf "Benchmarking is run for the first time. Creating reference input files with the results of these runs. \nIf any error occurs please delete the benchmarking directory and run the script after the values are corrected.\nCopying reference files...\n"
  cp -R testing_output $BmDir/Izh_sparse
  cp -R inputfiles10K $BmDir/Izh_sparse/inputfiles10K
else
  cp -R testing_output/testing.time $BmDir/Izh_sparse/testing.time
fi

#need to recompile if we want to rerun with different number of neurons. To be revisited...
#printf "\n\n*********************** Testing Izh_sparse 1K neurons****************************\n"
#printf "\n\n####################### Izh_sparse 1K GPU ######################\n"
#./generate_run 1 1000 1000 1 outdir Izh_sparse 0 0 1.0
#if [ "$firstrun" = true ]; then
#  cp -R inputfiles inputfiles1K
#fi 
#cp testing/testing.out.st testing/testing.out.st.GPU
#printf "\n\n####################### Izh_sparse 1K CPU ######################\n"
#./generate_run 0 1000 1000 1 outdir Izh_sparse 0 0
#cp testing/testing.out.st testing/testing.out.st.CPU

printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing PoissonIzh project ****************************\n"
printf "\n\n*******************************************************************************\n"

echo "Checking if benchmarking directory exists..."
if [ -d "$BmDir/PoissonIzh" ]; then
  echo "benchmarking directory exists. Using input data from" $BmDir/PoissonIzh 
  printf "\n"
  firstrun_PI=false;
else
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/PoissonIzh " and running the test only once (first run for reference)."
  mkdir -p $BmDir/PoissonIzh
  printf "\n"
  firstrun_PI=true;
fi 

cd ../PoissonIzh_project
make clean && make
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### PoissonIzh GPU test 1 ######################\n"
./generate_run 1 100 10 0.5 2 testing PoissonIzh 

printf "\n\n####################### PoissonIzh CPU test 1 ######################\n"
./generate_run 0 100 10 0.5 2 testing PoissonIzh 

cp -R testing_output/testing.time $BmDir/PoissonIzh/testing.time
if [ "$firstrun_PI" = true ]; then
  printf "Benchmarking is run for the first time. Creating reference input files with the results of these runs. \nIf any error occurs please delete the benchmarking directory and run the script after the values are corrected.\nCopying reference files...\n"
  cp -R ../PoissonIzh_project/testing_output $BmDir/PoissonIzh
 #other MB variants
else
  cp -R testing_output/testing.time $BmDir/PoissonIzh/testing.time
fi

printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing OneComp project ****************************\n"
printf "\n\n*******************************************************************************\n"
if [ ! -d "$BmDir/OneComp" ]; then
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/OneComp
  mkdir -p $BmDir/OneComp
  printf "\n"
fi 

cd ../OneComp_project
make clean && make
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### OneComp GPU test 1 ######################\n"
./generate_run 1 1 testing OneComp 

printf "\n\n####################### OneComp CPU test 1 ######################\n"
./generate_run 0 1 testing OneComp 

cp -R testing_output/testing.time $BmDir/OneComp/testing.time

printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing HHVclampGA project ****************************\n"
printf "\n\n*******************************************************************************\n"
if [ ! -d "$BmDir/HHVclampGA" ]; then
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/HHVclampGA
  mkdir -p $BmDir/HHVclampGA
  printf "\n"
fi

cd ../HHVclampGA_project
make clean && make
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing_output/testing.time
  printf "With new setup... \n"  >> testing_output/testing.time
fi
printf "\n\n####################### HHVclampGA GPU test 1 ######################\n"
./generate_run 1 2 5000 1000 testing 
printf "\n\n####################### HHVclampGA CPU test 1 ######################\n"
./generate_run 0 2 5000 1000 testing 

cp -R testing_output/testing.time $BmDir/HHVclampGA/testing.time

printf "\n\n*******************************************************************************\n"
printf "\n\n*********************** Testing SynDelay project ****************************\n"
printf "\n\n*******************************************************************************\n"
if [ ! -d "$BmDir/SynDelay" ]; then
  echo "Benchmarking directory does not exist. Creating a new one at" $BmDir/SynDelay
  mkdir -p $BmDir/SynDelay
  printf "\n"
fi

cd ../SynDelay_project
buildmodel.sh SynDelay
make clean && make release
if [ -d "testing_output" ]; then
  echo ${custommsg} >> testing.time
fi
printf "\n\n####################### SynDelay GPU test 1 ######################\n"
./syn_delay 1 testing
printf "\n\n####################### SynDelay CPU test 1 ######################\n"
./syn_delay 0 testing

cp -R testing_time $BmDir/SynDelay/testing_time

cd ..  


printf "\nTEST 2: TEST BY USING REFERENCE INPUT PATTERNS\n"  
#be careful with network sizes
#if you add new tests, don't forget to copy the output to your ref files 
cd MBody1_project
cp -R $BmDir/MBody1/* testing_output/
printf "With reference setup... \n"  >> testing_output/testing.time
printf "\n\n####################### MBody1 GPU TEST 2 ######################\n"
model/classol_sim testing 1
printf "\n\n####################### MBody1 CPU TEST 2 ######################\n"
model/classol_sim testing 0

cd ../MBody_individualID_project
#cp -R $BmDir/MBody_individualID/* testing_output/
cp -R $BmDir/MBody1/* testing_output/
printf "With reference setup... \n"  >> testing_output/testing.time
printf "\n\n####################### MBody_individualID GPU TEST 2 ######################\n"
model/classol_sim testing 1
printf "\n\n####################### MBody_individualID CPU TEST 2 ######################\n"
model/classol_sim testing 0

cd ../MBody_userdef_project
cp -R $BmDir/MBody1/* testing_output/
#cp -R $BmDir/MBody_userdef/* testing_output/
printf "With reference setup (same as MBody1 as well)... \n"  >> testing_output/testing.time
printf "\n\n####################### MBody_userdef GPU TEST 2 ######################\n"
model/classol_sim testing 1
printf "\n\n####################### MBody_userdef CPU TEST 2 ######################\n"
model/classol_sim testing 0

cd ../MBody_delayedSyn_project
cp -R $BmDir/MBody1/* testing_output/
#cp -R $BmDir/MBody_delayedSyn/* testing_output/
printf "With reference setup (same as MBody1 as well)...\n"  >> testing_output/testing.time
printf "\n\n####################### MBody_delayedSyn GPU TEST 2 ######################\n"
model/classol_sim testing 1
printf "\n\n####################### MBody_delayedSyn CPU TEST 2 ######################\n"
model/classol_sim testing 0

cd ../Izh_sparse_project
cp -R $BmDir/Izh_sparse/* testing_output/
printf "With reference setup (input is still random, so the results are not expected to be identical)... \n"  >> testing_output/testing.time
cp -R $BmDir/Izh_sparse/inputfiles10K/* inputfiles/
printf "\n\n####################### Izh_sparse GPU TEST 2 ######################\n"
model/Izh_sim_sparse testing 1
printf "\n\n####################### Izh_sparse CPU TEST 2 ######################\n"
model/Izh_sim_sparse testing 0

cd ../PoissonIzh_project
cp -R $BmDir/PoissonIzh/* testing_output/
printf "With reference setup... \n"  >> testing_output/testing.time
printf "\n\n####################### PoissonIzh GPU TEST 2 ######################\n"
model/PoissonIzh_sim testing 1
printf "\n\n####################### PoissonIzh CPU TEST 2 ######################\n"
model/PoissonIzh_sim testing 0

#Skipping OneComp and SynDelay, as test 2 does not make any sense for these project

cd ../HHVclampGA_project
cp -R $BmDir/HHVclampGA/* testing_output/
printf "With reference setup... \n"  >> testing_output/testing.time
printf "\n\n####################### HHVclampGA GPU TEST 2 ######################\n"
model/VClampGA testing 1 2
printf "\n\n####################### HHVclampGA CPU TEST 2 ######################\n"
model/VClampGA testing 0 2
cd ..
#cp -R $BmDir/PoissonIzh PoissonIzh_project/testing
#cp -R $BmDir/OneComp OneComp_project/testing
#cp -R $BmDir/SynDelay SynDelay_project/testing

cp MBody1_project/testing_output/testing.time $BmDir/MBody1/testing.time
cp MBody_userdef_project/testing_output/testing.time $BmDir/MBody_userdef/testing.time
cp Izh_sparse_project/testing_output/testing.time $BmDir/Izh_sparse/testing.time
cp PoissonIzh_project/testing_output/testing.time $BmDir/PoissonIzh/testing.time
cp OneComp_project/testing_output/testing.time $BmDir/OneComp/testing.time
cp HHVclampGA_project/testing_output/testing.time $BmDir/HHVclampGA/testing.time

printf "\nMBody1 time tail\n"
tail -n 18 MBody1_project/testing_output/testing.time
printf "\nMBody_individualID time tail\n"
tail -n 18 MBody_individualID_project/testing_output/testing.time
printf "\nMBody_userdef time tail\n"
tail -n 18 MBody_userdef_project/testing_output/testing.time
printf "\nMBody_delayedSyn time tail\n"
tail -n 18 MBody_delayedSyn_project/testing_output/testing.time
printf "\nIzh_sparse time tail\n"
tail -n 18 Izh_sparse_project/testing_output/testing.time
printf "\nPoissonIzh time tail\n"
tail -n 8 PoissonIzh_project/testing_output/testing.time
printf "\nOneComp time tail\n"
tail -n 8 OneComp_project/testing_output/testing.time  
printf "\nHHVclampGA time tail\n"
tail -n 8 HHVclampGA_project/testing_output/testing.time
printf "\nSynDelay time tail\n"
tail -n 8 SynDelay_project/testing_time

echo "Test complete! Checking if weekly copy of the output is necessary..."

monthandyear=`date +'%W_%m%y'`

echo ${monthandyear}
FILE="outputtestscript_"${monthandyear}
if [ -f $FILE ];
then
  echo $FILE" exists."
else
  cp outputtestscript $FILE
  echo "Created a copy of the output message at "${FILE}
fi
