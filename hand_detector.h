//
//  hand_detector.h
//  HandDemo
//
//  Created by Minghuang Ma on 10/22/14.
//
//

#ifndef HandDemo_hand_detector_h
#define HandDemo_hand_detector_h

#include "app_context.h"
#include "opencv2/opencv.hpp"

#include "ofMain.h"
#include "./HandDetector/HandDetector.hpp"
#include "./HandDetector/PrepData.h"

#include "train_async.h"

#include <vector>

class HandDetectorWrapper{
private:
    HandDetector hd_hand;
    HandDetector hd_obj;
    
    int knn;
    string root;
	string basename;
	
    // hand
    string img_prefix;
	string msk_prefix;
    string model_prefix;
    string feat_prefix;
    
    // object
    string obj_msk_prefix;
    string obj_img_prefix;
    string obj_model_prefix;
    string obj_feat_prefix;
	
    string feature_set;
    
    string grasp_mask_prefix;
    string grasp_img_prefix;
    string grasp_depth_prefix;
    
    int num_models;
    
    int sample_cnt;
    int num_train_imgs;
    
    int save_cnt;
    int num_grasp_imgs;
    
    cv::Mat hand_mask;
    
    std::vector<string> hand_filenames;
    std::vector<string> obj_filenames;
    
    PrepData prep_data;
    
    TrainAsync _trainAsync_hand;
    TrainAsync _trainAsync_obj;
    
public:
    void setup(AppContext *context);
    
    void start_sampling();
    int sample(AppContext *context);
    
    int train(AppContext *context);
    
    bool train_async(AppContext *context);
    
    void start_saving();
    int save(AppContext *context);
    
    void update(AppContext *context);
};


#endif
