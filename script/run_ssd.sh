#!/bin/bash

# Copyright (c) 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# 切换到内核态
setvars_path="/opt/intel/computer_vision_sdk_2018.5.445/bin/setupvars.sh"
source $setvars_path


# build_dir="$HOME/inference_engine_samples_build"

build_dir="/home/action/code/robocon/robocon_object_only/robocon_detection_ssd/build"

cd $build_dir

#./classification_sample -d $target -i $target_image_path -m "${ir_dir}/squeezenet1.1.xml" ${sampleoptions}

# MYRID only support FP16
#./robocon_gui -m=/home/action/code/caffe/caffe_tool/optmi_FP16/caffenet_train_iter_2500.xml -dg=true -d=MYRIAD

# CPU only support FP16
#./robocon_gui -m=/home/action/code/caffe/caffe_tool/optmi_FP32/caffenet_train_iter_2500.xml -dg=true -d=CPU

#./robocon_gui -m=/home/action/code/caffe/caffe_tool/optmi/caffenet_train_iter_2500.xml 
#./text_detection_demo -i=/opt/intel/computer_vision_sdk_2018.5.445/deployment_tools/intel_models/text-detection-0001/description/text-detection-0001.png -m=/opt/intel/computer_vision_sdk_2018.5.445/deployment_tools/intel_models/text-detection-0001/FP16/text-detection-0001.xml -d MYRIAD

./robocon_detection_ssd -m=/home/action/code/caffe/model/robocon_ssd300/optmi_FP16/mobilenet_iter_35118.xml -dg=false -d=MYRIAD


printf "${dashes}"
printf "Demo completed successfully.\n\n"

