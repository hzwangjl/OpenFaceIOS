Cmake_minimum_required(VERSION 2.8)
project(OpenFaceLinux)

SET(FA_SRC "/home/hzwangjialei/project/OpenFaceIOS/FaceAR_SDK_IOS_OpenFace_RunFull/lib/local/FaceAnalyser/src")
SET(FA_INC "/home/hzwangjialei/project/OpenFaceIOS/FaceAR_SDK_IOS_OpenFace_RunFull/lib/local/FaceAnalyser/include")

SET(GE_SRC "/home/hzwangjialei/project/OpenFaceIOS/FaceAR_SDK_IOS_OpenFace_RunFull/lib/local/GazeEstimate/")
SET(GE_INC "/home/hzwangjialei/project/OpenFaceIOS/FaceAR_SDK_IOS_OpenFace_RunFull/lib/local/GazeEstimate/")

SET(LD_SRC "/home/hzwangjialei/project/OpenFaceIOS/FaceAR_SDK_IOS_OpenFace_RunFull/lib/local/LandmarkDetector/src")
SET(LD_INC "/home/hzwangjialei/project/OpenFaceIOS/FaceAR_SDK_IOS_OpenFace_RunFull/lib/local/LandmarkDetector/include")

SET(OPENCV_LIB "/home/hzwangjialei/commlibs/opencv-3.4.1/lib")
SET(OPENCV_INC "/home/hzwangjialei/commlibs/opencv-3.4.1/include")

SET(DLIB_LIB "/home/hzwangjialei/commlibs/dlib")
SET(DLIB_INC "/home/hzwangjialei/commlibs/dlib/include")

SET(TBB_LIB "/home/hzwangjialei/commlibs/tbb/lib")
SET(TBB_INC "/home/hzwangjialei/commlibs/tbb/include")

#SET(OPENCV_LIB "/home/hzwangjialei/commlibs/anaconda2/pkgs/opencv-3.1.0-np112py27_1/lib")
#SET(OPENCV_INC "/home/hzwangjialei/commlibs/anaconda2/pkgs/opencv-3.1.0-np112py27_1/include")

file(GLOB CPP_SRCS "${FA_SRC}/*.cpp" "${GE_SRC}/*.cpp" "${LD_SRC}/*.cpp")
file(GLOB CPP_INCS "${FA_INC}/*.h"   "${GE_INC}/*.h"   "S{LD_INC}/*.h" )
file(GLOB TEST_SRCS "test/test.cpp")

INCLUDE_DIRECTORIES(${OPENCV_INC} ${FA_INC} ${GE_SRC} ${LD_INC} ${DLIB_INC} ${TBB_INC})
LINK_DIRECTORIES(${OPENCV_LIB} ${TBB_LIB})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -g -O2 -Wall -Wno-sign-compare -D__STDC_CONSTANT_MACROS" )

add_definitions(${DEFINE_FLAGS})

add_executable(openface ${TEST_SRCS} ${CPP_SRCS})
target_link_libraries(openface -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lopencv_objdetect)
