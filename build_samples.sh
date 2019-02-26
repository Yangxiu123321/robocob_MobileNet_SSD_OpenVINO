#!/bin/bash


# build_dir=$HOME/inference_engine_robocon_build
build_dir=/home/action/code/robocon/robocon_object/bin
CMAKELISTS_DIR=/home/action/code/robocon/robocon_object
#添加-p参数设置CMAKE_CURRENT_BINARY_DIR为build_dir
mkdir -p $build_dir
cd $build_dir
cmake -D CMAKE_BUILD_TYPE=Release $CMAKELISTS_DIR
make -j8

printf "\nBuild completed, you can find binaries for all samples in the $build_dir/intel64/Release subfolder.\n\n"

# /opt/intel/computer_vision_sdk_2018.5.445/inference_engine/samples
