#include <stdio.h>

#if 0
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

using namespace std;

LandmarkDetector::FaceModelParameters det_parameters;
LandmarkDetector::CLNF clnf_model;

void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image, const LandmarkDetector::CLNF& face_model,
        const LandmarkDetector::FaceModelParameters& det_parameters, int frame_count, double fx, double fy, double cx, double cy)
{
    // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
    double detection_certainty = face_model.detection_certainty;
    bool detection_success = face_model.detection_success;

    double visualisation_boundary = 0.2;

    // Only draw if the reliability is reasonable, the value is slightly ad-hoc
    if (detection_certainty < visualisation_boundary)
    {
        LandmarkDetector::Draw(captured_image, face_model);

        double vis_certainty = detection_certainty;
        if (vis_certainty > 1)
            vis_certainty = 1;
        if (vis_certainty < -1)
            vis_certainty = -1;

        vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

        // A rough heuristic for box around the face width
        int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

        cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

        // Draw it in reddish if uncertain, blueish if certain
        LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);
        LandmarkDetector::ShowActionUnits(captured_image);
    }
}

int main(int argc, char** argv)
{
    det_parameters.init();
    det_parameters.model_location = "../../FaceAR_SDK_IOS_OpenFace_RunFull/model/model/main_clnf_general.txt";
    det_parameters.face_detector_location = "../../FaceAR_SDK_IOS_OpenFace_RunFull/model/classifiers/haarcascade_frontalface_alt.xml";

    clnf_model.model_location_clnf = "../../FaceAR_SDK_IOS_OpenFace_RunFull/model/model/main_clnf_general.txt";
    clnf_model.face_detector_location_clnf = "../../FaceAR_SDK_IOS_OpenFace_RunFull/model/classifiers/haarcascade_frontalface_alt.xml";
    clnf_model.inits();

    std::cout << "model_location = " << det_parameters.model_location << std::endl;
    std::cout << "face_detector_location = " << det_parameters.face_detector_location << std::endl;

    cv::Mat img = cv::imread("../test.jpg");
    //cv::IMREAD_GRAYSCALE);

    cv::Mat_<float> depth_image;
    cv::Mat_<uchar> grayscale_image;

    cvtColor(img, grayscale_image, cv::COLOR_BGR2GRAY);

    bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);

    cout << "detection_success = " << detection_success << endl;

    double detection_certainty = clnf_model.detection_certainty;

    int frame_count;
    float fx, fy, cx, cy;
    cx = 1.0 * img.cols / 2.0;
    cy = 1.0 * img.rows / 2.0;

    fx = 500 * (img.cols / 640.0);
    fy = 500 * (img.rows / 480.0);

    fx = (fx + fy) / 2.0;
    fy = fx;
    visualise_tracking(img, depth_image, clnf_model, det_parameters, frame_count, fx, fy, cx, cy);

    cv::Point3f gazeDirection0(0, 0, -1);
    cv::Point3f gazeDirection1(0, 0, -1);
    if (det_parameters.track_gaze && detection_success && clnf_model.eye_model){
        GazeEstimate::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
        GazeEstimate::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
        GazeEstimate::DrawGaze(img, clnf_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
    }

    cv::imwrite("./aa.jpg", img);
    system("~/imgcat.sh aa.jpg");

    printf("hahah \n");
}
