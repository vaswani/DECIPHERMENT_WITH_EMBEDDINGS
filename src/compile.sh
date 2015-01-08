#! /bin/bash

/opt/local/bin/g++-mp-4.7 -c -I/Users/avaswani/Research/GITHUB/NEURAL_LANGUAGE_MODEL_ISI/3rdparty/ -I/opt/local/include/ -L/opt/local/lib -fopenmp  testMappingOptimization.cpp
/opt/local/bin/g++-mp-4.7 -c  -I/Users/avaswani/Research/GITHUB/NEURAL_LANGUAGE_MODEL_ISI/3rdparty/ -I/opt/local/include/ -L/opt/local/lib -fopenmp  util.cpp

/opt/local/bin/g++-mp-4.7 -o testMappingOptimization testMappingOptimization.o util.o -L/opt/local/lib -fopenmp
