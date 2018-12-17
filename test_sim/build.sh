g++ -std=c++11 simulator.cc -o simulator -L../generated_code -L$CUDA_PATH/lib64 -I../generated_code -I$CUDA_PATH/include -lcuda -lcudart -lrunner -Wl,-rpath=../generated_code
