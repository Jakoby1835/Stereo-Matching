#!/bin/bash

echo "Compiling StereoMatcher.cpp ..."
mpic++ StereoMatcher.cpp -o StereoMatcher -fopenmp -lOpenCL
echo "Done Compiling StereoMatcher"
