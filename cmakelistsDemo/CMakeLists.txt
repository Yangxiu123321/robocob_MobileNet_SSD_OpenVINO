cmake_minimum_required(VERSION 2.8)

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lxml2")
####################################

set (GFLAGS_IS_SUBPROJECT TRUE)
set (HAVE_SYS_STAT_H 1)
set (HAVE_INTTYPES_H 1)

set (TARGET_NAME "robocon_detection_ssd")

# 设置成release
if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

#include
set (SERIAL_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/serial/src)
set (CAMERA_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/MindVisionAPI/src)
set (OPENVINO_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/inference_engine/src)

#third ib
set (gflags_lib ${CMAKE_CURRENT_SOURCE_DIR}/gflags/lib/libgflags_nothreads.a)
set (format_reader_lib ${CMAKE_CURRENT_SOURCE_DIR}/common/format_reader/lib/libformat_reader.so)

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        )

file(GLOB SERIAL_SRC ${SERIAL_DIR}/*.cc)
file(GLOB CAMERA_SRC ${CAMERA_DIR}/*.cpp)
file(GLOB OPENVINO_SRC ${OPENVINO_DIR}/*.cpp)



# use this flag if you need to throw custom message in case if the IE package is not found.
set (InferenceEngine_DIR /opt/intel/computer_vision_sdk/inference_engine/share)
# set (InferenceEngine_DIR /home/action/code/robocon/robocon_object_only/robocon_detection_ssd/inference_engine/share)
if (IE_NOT_FOUND_MESSAGE)
    find_package(InferenceEngine 1.5 QUIET)
    if (NOT(InferenceEngine_FOUND))
        message(FATAL_ERROR ${IE_NOT_FOUND_MESSAGE})
    endif()
else()
    #找到
    find_package(InferenceEngine 1.5 REQUIRED)
endif()

if (UNIX)
    set (LIB_DL dl)
endif()


# Find OpenCV library if exists
set (OpenCV_DIR "/home/action/code/opencv/opencv-3.4.5/build")
find_package(OpenCV 3.4.5 REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_definitions(-DUSE_OPENCV)
else()
    set (BUILD_VALIDATION_APP OFF)
    message(WARNING "No suitable OpenCV version detected, BUILD_VALIDATION_APP is set to OFF")
endif()


# serial lib
add_library(serial SHARED ${SERIAL_SRC})
install(TARGETS serial LIBRARY DESTINATION lib)

#camear lib
add_library(camera SHARED ${CAMERA_SRC})
target_link_libraries(camera ${CMAKE_CURRENT_SOURCE_DIR}/MindVisionAPI/Library/libMVSDK.so)
install(TARGETS camera LIBRARY DESTINATION lib)

#openvino lib
add_library(openvino SHARED ${OPENVINO_SRC})
target_link_libraries(openvino IE::ie_cpu_extension ${InferenceEngine_LIBRARIES})
install(TARGETS openvino LIBRARY DESTINATION lib)

include_directories(
        #serial
        ${SERIAL_DIR}
        # openvino
        ${OPENVINO_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/format_reader
        ${InferenceEngine_INCLUDE_DIRS}
        ${InferenceEngine_INCLUDE_DIRS}/../src/extension
        ${CMAKE_CURRENT_SOURCE_DIR}/gflags/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        #camera
        ${CMAKE_CURRENT_SOURCE_DIR}/MindVisionAPI/include
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})
target_link_libraries(
    ${TARGET_NAME}
    serial
    camera
    openvino
    ${format_reader_lib} 
    ${gflags_lib} 
    ${OpenCV_LIBS})

if(UNIX)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread)
endif()
