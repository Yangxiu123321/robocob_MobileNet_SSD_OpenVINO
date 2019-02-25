#include "openvino.hpp"
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <gflags/gflags.h>

using namespace InferenceEngine;

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to an .bmp image.";

/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the sample will look for this plugin only";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. " \
"Sample will look for a suitable plugin for device specified";

/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report";

/// @brief message for iterations count
static const char iterations_count_message[] = "Number of iterations (default 1)";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels. "\
"Absolute path to the xml file with the kernels desc.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers. " \
"Absolute path to a shared library with the kernels impl.";

/// @brief message for plugin messages
static const char plugin_err_message[] = "Enables messages from a plugin";

/// @brief message for com
static const char com_message[] = "show the com information(default is /dev/ttyUSB0)";

/// @brief message for debug
static const char debug_message[] = "show the debug information(default is true)";


/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief state you com<br>
DEFINE_string(com, "/dev/ttyUSB0", com_message);

/// @brief Define flag for debug<br>
DEFINE_bool(dg, true, debug_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// \brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// \brief Define parameter for set path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, "", plugin_path_message);

/// \brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Iterations count (default 1)
DEFINE_int32(ni, 1, iterations_count_message);

/// @brief Enable plugin messages
DEFINE_bool(p_msg, false, plugin_err_message);

/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "object_detection_sample_ssd [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -pp \"<path>\"            " << plugin_path_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -pc                     " << performance_counter_message << std::endl;
    std::cout << "    -ni \"<integer>\"         " << iterations_count_message << std::endl;
    std::cout << "    -p_msg                  " << plugin_err_message << std::endl;
}


ConsoleErrorListener error_listener;

// #include "object_detection_sample_ssd.h"
 OPENVINO_STATUS ParseAndCheckCommandLine(int argc, char *argv[]) {
     OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return status;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        std::cout << "[ERROR]" << "-m not set" << std::endl; 
        return status;
    }

    return status;
}

Openvino::Openvino(int argc, char *argv[])
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    status = ParseAndCheckCommandLine(argc,argv);
    if(status != OPENVINO_STATUS::SUCESS)
    {
        std::cout << "[ERROR]" << "ParseAndCheckCommandLine" << std::endl;
        // return status;
    }

    status = init();
    if(status != OPENVINO_STATUS::SUCESS)
    {
        std::cout << "[ERROR]" << "init" << std::endl;
        // return status;
    }
    // return status;
}

Openvino::~Openvino()
{
    
}

OPENVINO_STATUS Openvino::loadPlugin(void)
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    // --------------------------- 3. Load Plugin for inference engine -------------------------------------
    slog::info << "Loading plugin" << slog::endl;
    plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64" , "" }).getPluginByDevice(FLAGS_d);
    if (FLAGS_p_msg) {
        static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
    }

    /*If CPU device, load default library with extensions that comes with the product*/
    if (FLAGS_d.find("CPU") != std::string::npos) {
        /**
        * cpu_extensions library is compiled from "extension" folder containing
        * custom MKLDNNPlugin layer implementations. These layers are not supported
        * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    if (!FLAGS_l.empty()) {
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
        plugin.AddExtension(extension_ptr);
        slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
    }

    if (!FLAGS_c.empty()) {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
        slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
    }

    /** Setting plugin parameter for per layer metrics **/
    if (FLAGS_pc) {
        plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
    }

    /** Printing plugin version **/
    printPluginVersion(plugin, std::cout);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
    slog::info << "Loading network files:"
        "\n\t" << FLAGS_m <<
        "\n\t" << binFileName <<
        slog::endl;

    /** Read network model **/
    networkReader.ReadNetwork(FLAGS_m);

    /** Extract model name and load weights **/
    networkReader.ReadWeights(binFileName);
    network = networkReader.getNetwork();
    // -----------------------------------------------------------------------------------------------------
    return status;
}

OPENVINO_STATUS Openvino::prepareInputBlobs(void)
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    // --------------------------- 5. Prepare input blobs --------------------------------------------------
    slog::info << "Preparing input blobs" << slog::endl;

    /** Taking information about all topology inputs **/
    InputsDataMap inputsInfo(network.getInputsInfo());

    /** SSD network has one input and one output **/
    if (inputsInfo.size() != 1 && inputsInfo.size() != 2) throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

    /**
     * Some networks have SSD-like output format (ending with DetectionOutput layer), but
     * having 2 inputs as Faster-RCNN: one for image and one for "image info".
     *
     * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
     * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
     */


    // InputInfo::Ptr inputInfo = inputsInfo.begin()->second;

    // SizeVector inputImageDims;
    /** Stores input image **/

    /** Iterating over all input blobs **/
    for (auto & item : inputsInfo) {
        /** Working with first input tensor that stores image **/
        if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
            imageInputName = item.first;

            slog::info << "Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;

            /** Creating first input blob **/
            Precision inputPrecision = Precision::U8;
            item.second->setPrecision(inputPrecision);
        } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
            imInfoInputName = item.first;

            Precision inputPrecision = Precision::FP32;
            item.second->setPrecision(inputPrecision);
            if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6) ||
                    item.second->getTensorDesc().getDims()[0] != 1) {
                std::cout << "[ERROR]" << "Invalid input info. Should be 3 or 6 values length" << std::endl;                
                return status;
                // throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
            }
        }
    }
    // -----------------------------------------------------------------------------------------------------
    return status;
}

OPENVINO_STATUS Openvino::prepareoutputBlobs(void)
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    // --------------------------- 6. Prepare output blobs -------------------------------------------------
    slog::info << "Preparing output blobs" << slog::endl;

    OutputsDataMap outputsInfo(network.getOutputsInfo());

    DataPtr outputInfo;
    for (const auto& out : outputsInfo) {
        if (out.second->creatorLayer.lock()->type == "DetectionOutput") {
            outputName = out.first;
            outputInfo = out.second;
        }
    }

    if (outputInfo == nullptr) {
        std::cout << "[ERROR]" << "Can't find a DetectionOutput layer in the topology" << std::endl;
        // throw std::logic_error("Can't find a DetectionOutput layer in the topology");
        return status;
    }

    outputDims = outputInfo->getTensorDesc().getDims();
    maxProposalCount = outputDims[2];
    objectSize = outputDims[3];

    if (objectSize != 7) {
        // throw std::logic_error("Output item should have 7 as a last dimension");
        std::cout << "[ERROR]" << "Output item should have 7 as a last dimension" << std::endl;
        return status;
    }

    if (outputDims.size() != 4) {
        std::cout << "[ERROR]" << "Incorrect output dimensions for SSD model" << std::endl;        
        return status;
        // throw std::logic_error("Incorrect output dimensions for SSD model");
    }

    /** Set the precision of output data provided by the user, should be called before load of the network to the plugin **/
    outputInfo->setPrecision(Precision::FP32);
    // -----------------------------------------------------------------------------------------------------
    return status;
}

OPENVINO_STATUS Openvino::createInferenceEngine(void)
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    // --------------------------- 7. Loading model to the plugin ------------------------------------------
    slog::info << "Loading model to the plugin" << slog::endl;

    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Create infer request -------------------------------------------------
    infer_request = executable_network.CreateInferRequest();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 9. Prepare input --------------------------------------------------------
    /** Collect images data ptrs **/

    batchSize = network.getBatchSize();
    slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;
    if(batchSize != 1)
    {
        slog::warn << "Batch size is " << std::to_string(batchSize) << slog::endl;
    }
    return status;
}


/*
OPENVINO_STATUS Openvino::judgeStill(void)
{
    // --------------------------- 6. bsg --------------------------------------------------------
    cv::Mat grayImg;
    cv::cvtColor(resized,grayImg,6);
    // imshow("gray",grayImg);

    cv::Mat dstImg;
    grayImg.copyTo(dstImg,mask);
    // imshow("dst",dstImg);

    cv::Mat bgmask;
    bgsubtractor->apply(dstImg,bgmask,0.05);
    cv::imshow("bmk",bgmask);

    int redMoreBlueNum = 0;
    int blueMoreRedNum = 0;
    // 记录R>B
    for (int i = 0;i<RESIZE_IMG_HEIGHT;i++)
    {
        uchar* bgmaskData = bgmask.ptr<uchar>(i);
        for(int j = 0;j<RESIZE_IMG_WIDTH;j++)
        {
            uchar* srcData = resized.ptr<uchar>(i,j);
            // 计数需要满足的条件://1、需要为白色，白色的地方说明物体在动。2、R > G(G > R)。3、总体分量比较小，并且需要的分量最大
            if( (bgmaskData[j]) && (srcData[0] > srcData[2]) && (srcData[0] > srcData[1]) && ((srcData[0] < 100) && (srcData[1] < 70) && (srcData[2] < 70)))
            {
                blueMoreRedNum ++;
            }else if( (bgmaskData[j]) && (srcData[2] > srcData[0]) && (srcData[2] > srcData[1]) && ((srcData[0] < 70) && (srcData[1] < 70) && (srcData[2] < 100)))
            {
                redMoreBlueNum ++;
            }else
            {
                continue;
            }
        }
    }

    if (FLAGS_dg)
    {
        cv::imshow("bmk",bgmask);
    }else{

    }
    // std::cout << redMoreBlueNum << " " << blueMoreRedNum << std::endl;
    if(FLAGS_ch && (redMoreBlueNum < 200))
    {
        still_num ++;
        if(still_num > 2)
        {
            step = JUDGE_CLASSES;
            still_num = 0;
        }
    }else if(!FLAGS_ch && (blueMoreRedNum < 200))
    {
        still_num ++;
        if(still_num > 2)
        {
            step = JUDGE_CLASSES;
            still_num = 0;
        }
    }else
    {
        still_num = 0;
    }
}
*/

OPENVINO_STATUS Openvino::judgeClass(void)
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    cv::Mat debugImg;
    srcImg.copyTo(debugImg);
    
    cv::resize(debugImg, debugImg, cv::Size(640, 480));
    imshow("src",debugImg);

    // resized image
    cv::resize(srcImg, resized, cv::Size(RESIZE_IMG_WIDTH, RESIZE_IMG_HEIGHT));
    size_t size = resized.size().width * resized.size().height * resized.channels();
    imagesData.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

    // store iamge data to imagesData
    for (size_t id = 0; id < size; ++id) {
        imagesData.get()[id] = resized.data[id];
    }
    imageWidths = 640;
    imageHeights = 480;
    if (imagesData.get() == nullptr)
    {
        std::cout << "[ERROR]" << "imagesData.get error" << std::endl;
        return status;
    }


    /** Creating input blob **/
    Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

    /** Filling input tensor with images. First b channel, then g and r channels **/
    size_t num_channels = imageInput->getTensorDesc().getDims()[1];
    size_t image_size = imageInput->getTensorDesc().getDims()[3] * imageInput->getTensorDesc().getDims()[2];

    unsigned char* data = static_cast<unsigned char*>(imageInput->buffer());

    /** Iterate over all input images **/
    /** Iterate over all pixel in image (b,g,r) **/
    for (size_t pid = 0; pid < image_size; pid++) {
        /** Iterate over all channels **/
        for (size_t ch = 0; ch < num_channels; ++ch) {
            /**          [images stride + channels stride + pixel id ] all in bytes            **/
            data[ch * image_size + pid] = imagesData.get()[pid*num_channels + ch];
        }
    }
// --------------------------- 10. Do inference ---------------------------------------------------------
    infer_request.Infer();
// --------------------------- 11. Process output -------------------------------------------------------
    const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
    const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

    //std::vector<std::vector<int> > boxes(batchSize);
    std::vector<std::vector<int> > classes(batchSize);

    /* Each detection has image_id that denotes processed image */
    // maxProposalCount is 200
    // 计算分数
    int score = 0;
    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
        float image_id = detection[curProposal * objectSize + 0];
        if (image_id < 0) {
            break;
        }

        float label = detection[curProposal * objectSize + 1];
        float confidence = detection[curProposal * objectSize + 2];
        float xmin = detection[curProposal * objectSize + 3] * imageWidths;
        float ymin = detection[curProposal * objectSize + 4] * imageHeights;
        float xmax = detection[curProposal * objectSize + 5] * imageWidths;
        float ymax = detection[curProposal * objectSize + 6] * imageHeights;
        
        // std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
        //     "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id;

        if (confidence > 0.5) {
            /** Drawing only objects with >50% probability **/
            // classes[image_id].push_back(static_cast<int>(label));
            // boxes[image_id].push_back(static_cast<int>(xmin));
            // boxes[image_id].push_back(static_cast<int>(ymin));
            // boxes[image_id].push_back(static_cast<int>(xmax - xmin));
            // boxes[image_id].push_back(static_cast<int>(ymax - ymin));
            
            // draw retangle
            std::cout << label << " WILL BE PRINTED!" << "," << "confidence is:" << confidence << std::endl;;
            cv::rectangle(debugImg,cv::Rect2f(cv::Point(xmin,ymin),cv::Point(xmax,ymax)),cv::Scalar(255,0,255),1);
            imshow("objectDecetion",debugImg);
            // 判断得分
            switch((int)label)
            {
                case 1:
                    score += 50;
                break;
                case 2:
                    score += 40;
                break;
                case 3:
                    score += 20;
                break;

                default:
                break;
            }
        }
    }
    std::cout << "score is " << score << std::endl;
// -----------------------------------------------------------------------------------------------------

    /** Show performance results **/
    if (FLAGS_pc) {
        printPerformanceCounts(infer_request, std::cout);
    }
    return status;
}

OPENVINO_STATUS Openvino::init(void)
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    status = loadPlugin();
    if(status != OPENVINO_STATUS::SUCESS)
    {
        std::cout << "[ERROR]" << "loadPlugin error" << std::endl;
        return status;
    }

    status = prepareInputBlobs();
    if(status != OPENVINO_STATUS::SUCESS)
    {
        std::cout << "[ERROR]" << "prepareInputBlobs error" << std::endl;
        return status;
    }

    status = prepareoutputBlobs();
    if(status != OPENVINO_STATUS::SUCESS)
    {
        std::cout << "[ERROR]" << "prepareoutputBlobs error" << std::endl;
        return status;
    }

    status = createInferenceEngine();
    if(status != OPENVINO_STATUS::SUCESS)
    {
        std::cout << "[ERROR]" << "createInferenceEngine error" << std::endl;
        return status;
    }
    return status;
}

OPENVINO_STATUS Openvino::inference(void)
{
    OPENVINO_STATUS status = OPENVINO_STATUS::SUCESS;

    status = judgeClass();
    if(status != OPENVINO_STATUS::SUCESS)
    {
        std::cout << "[ERROR]" << "judgeClass error" << std::endl;
        return status;
    }

    // status = sendMessage();
    // if(status != OPENVINO_STATUS::SUCESS)
    // {
    //     std::cout << "[ERROR]" << "sendMessage error" << std::endl;
    //     return status;
    // }
    return status;
}
