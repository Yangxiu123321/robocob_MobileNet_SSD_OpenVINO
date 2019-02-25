#ifndef __OPENVINO_H
#define __OPENVINO_H

#include <random>
#include <string>
#include <memory>
#include <vector>
#include <time.h>
#include <limits>
#include <chrono>
#include <iostream>
#include <algorithm>

#include <format_reader_ptr.h>
#include <inference_engine.hpp>
#include <ext_list.hpp>

#include "opencv2/opencv.hpp"


#define BBOX_THICKNESS 2

#define RESIZE_IMG_WIDTH 300
#define RESIZE_IMG_HEIGHT 300

using namespace InferenceEngine;

enum OPENVINO_STATUS{SUCESS = 0,FAIL};
// bone get  score
enum BONE_SCORE{BACKGROUND = 0,CAMEL,HORSE,CLIMP_UP,OUT};
class Openvino
{
public:
    /*openvino*/
    Openvino(int argc, char *argv[]);
    ~Openvino();
    OPENVINO_STATUS loadPlugin(void); 
    OPENVINO_STATUS prepareInputBlobs(void);
    OPENVINO_STATUS prepareoutputBlobs(void);
    OPENVINO_STATUS createInferenceEngine(void);

    // OPENVINO_STATUS judgeStill(void);
    OPENVINO_STATUS judgeClass(void);

    //init
    OPENVINO_STATUS init(void);
    OPENVINO_STATUS inference(void);


    cv::Mat srcImg;
    char runSignal;

private:

    std::string imageInputName, imInfoInputName;
    std::string outputName;
    int objectSize;
    SizeVector outputDims;
    int maxProposalCount;

    InferencePlugin plugin;

    CNNNetReader networkReader;

    CNNNetwork network;

    InferRequest infer_request;

    std::shared_ptr<unsigned char> imagesData, originalImagesData;
    int imageWidths, imageHeights;
    size_t batchSize;
    cv::Mat resized;
    /* data */
};//openvino inference

#endif