//
//  FaceARDetectIOS.m
//  FaceAR_SDK_IOS_OpenFace_RunFull
//
//  Created by Keegan Ren on 7/5/16.
//  Copyright © 2016 Keegan Ren. All rights reserved.
//

#import "FaceARDetectIOS.h"

LandmarkDetector::FaceModelParameters det_parameters;
// The modules that are being used for tracking
LandmarkDetector::CLNF clnf_model;

FaceAnalysis::FaceAnalyser face_analyser;

@implementation FaceARDetectIOS

//bool inits_FaceAR();
-(id) init
{
    self = [super init];
    NSString *location = [[NSBundle mainBundle] resourcePath];
    det_parameters.init();
    det_parameters.model_location = [location UTF8String] + std::string("/model/main_clnf_general.txt");
    det_parameters.face_detector_location = [location UTF8String] + std::string("/classifiers/haarcascade_frontalface_alt.xml");

    std::cout << "model_location = " << det_parameters.model_location << std::endl;
    std::cout << "face_detector_location = " << det_parameters.face_detector_location << std::endl;

    clnf_model.model_location_clnf = [location UTF8String] + std::string("/model/main_clnf_general.txt");
    clnf_model.face_detector_location_clnf = [location UTF8String] + std::string("/classifiers/haarcascade_frontalface_alt.xml");
    clnf_model.inits();

    face_analyser.model_path = [location UTF8String] + std::string("/AU_predictors/");
    face_analyser.svm_model_path = [location UTF8String] + std::string("/AU_predictors/main_static_svms.txt");
    face_analyser.init();

    return self;
}

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, cv::Mat_<float>& depth_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, int frame_count, double fx, double fy, double cx, double cy)
//-(BOOL) visualise_tracking:(cv::Mat)captured_image depth_image_:(cv::Mat)depth_image face_model_:(const LandmarkDetector::CLNF)face_model det_parameters_:(const LandmarkDetector::FaceModelParameters)det_parameters frame_count_:(int)frame_count fx_:(double)fx fy_:(double)fy cx_:(double)cx cy_:(double)cy
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
    }
}

void ShowActionUnits(std::vector<std::pair<std::string, double> > au_class, cv::Mat& image)
{
    CvScalar color;
    //std::cout << "ShowActionUnits " << au_class.size() << std::endl;
    for(int i = 0; i < au_class.size(); i++) {
        if(au_class[i].second == 1){
            color = cvScalar(0, 0, 255);
        }
        else{
            color = cvScalar(0, 255, 0);
        }
        putText(image, au_class[i].first, cvPoint(10, 20 + 25 * i), 1, 1.5, color, 2);
    }
}


//bool run_FaceAR(cv::Mat &captured_image, int frame_count, float fx, float fy, float cx, float cy);
-(BOOL) run_FaceAR:(cv::Mat)captured_image frame__:(int)frame_count fx__:(double)fx fy__:(double)fy cx__:(double)cx cy__:(double)cy
{
    // Reading the images
    cv::Mat_<float> depth_image;
    cv::Mat_<uchar> grayscale_image;

    if(captured_image.channels() == 3)
    {
        cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
    }
    else
    {
        grayscale_image = captured_image.clone();
    }

    // The actual facial landmark detection / tracking
    bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_model, det_parameters);
    //            bool detection_success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, clnf_model, det_parameters);

    // Visualising the results
    // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
    double detection_certainty = clnf_model.detection_certainty;

    visualise_tracking(captured_image, depth_image, clnf_model, det_parameters, frame_count, fx, fy, cx, cy);

    /********** gaze EstimateGaze **********/
#if 0
    cv::Point3f gazeDirection0(0, 0, -1);
    cv::Point3f gazeDirection1(0, 0, -1);
    if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
    {
        GazeEstimate::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
        GazeEstimate::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
        GazeEstimate::DrawGaze(captured_image, clnf_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
    }
#endif

    /********** AU Detections **********/
    if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAURegNames().size()) {
        std::cout << "WARNING: no Action Unit models found" << std::endl;
    }

    cv::Mat sim_warped_img;
    cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;
    if(detection_success) {
        face_analyser.PredictStaticAUsAndComputeFeatures(captured_image, clnf_model.detected_landmarks);
        face_analyser.GetLatestAlignedFace(sim_warped_img);
        face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
    }

    std::vector<std::pair<std::string, double> > au_regs = face_analyser.GetCurrentAUsReg();
    std::vector<std::pair<std::string, double> > au_class = face_analyser.GetCurrentAUsClass();

    ShowActionUnits(au_class, captured_image);

    return true;
}

//bool reset_FaceAR();
-(BOOL) reset_FaceAR
{
    clnf_model.Reset();

    return true;
}

//bool clear_FaceAR();
-(BOOL) clear_FaceAR
{
    clnf_model.Reset();

    return true;
}


@end
