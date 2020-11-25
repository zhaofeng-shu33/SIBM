#!/bin/bash
# build all c++ files
g++ test.cpp -lgtest -lgtest_main -lpthread -o build/test
g++ sibm_experiment.cpp -o build/sibm_experiment
python3 setup.py build_ext --inplace
