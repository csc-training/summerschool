#!/bin/bash

./build.sh || { echo "Failed to build"; exit 1; }
ctest --test-dir ctest-gtest/build/Release -j 6
