// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// COM
#include <cstdio>
#include <unistd.h>
#include "serial.h"

#include "mv_init.h"
#include "openvino.hpp"

using namespace cv;


typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

int main(int argc, char *argv[]){    
    // serial Init
    serial::Serial my_serial("/dev/ttyUSB0", 115200, serial::Timeout::simpleTimeout(2));
    if(my_serial.isOpen())
    {
        std::cout << "[INFO]" << "serial port initialize ok" << std::endl;
    }else{
        std::cout << "[ERROR]" << "can't find serial" << std::endl;
        return -1;
    }

    // choose blue(1) or red(0) playground 
    int palygroundId = 0;
    // mindVersion camera initilize
    MvInit mvCamera(palygroundId%2);

    // openVINO initilize
    Openvino openvino(argc,argv);

    while(1)
    {
      auto t0 = Time::now();
      // image size is 1280*960
      openvino.srcImg = mvCamera.getImage();

      // openvino do inference
      openvino.inference();

      // performance
      double total = 0;
      auto t1 = Time::now();
      fsec fs = t1 - t0;
      ms d = std::chrono::duration_cast<ms>(fs);
      total = d.count();
      std::cout << "total inference time: " << total << std::endl;
      std::cout << "Throughput: " << 1000 / total << " FPS" << std::endl;
      std::cout << std::endl;

      int c = cv::waitKey(1);
      if(c == 27 || c == 'q' || c == 'Q')
      {
        break;
      }
    }
}