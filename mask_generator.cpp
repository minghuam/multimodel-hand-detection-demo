//
//  mask_generator.cpp
//  HandDemo
//
//  Created by Minghuang Ma on 10/23/14.
//  Port Akansha's code to OpenCV 2.4.9
//

#include "mask_generator.h"
#include <iostream>

void MaskGenerator::setup(AppContext *context){
    erosion_size = 3;
    dilation_size = 1;
    max_learning_frames = 30;
    learning_frame = 0;
}

/*
    learn background model and return true when done
*/
bool MaskGenerator::learn(AppContext *context){
    
    const int max_int_frames = 2;
    static int int_frame_cnt = 0;
    
    if(learning_frame >= max_learning_frames){
        return true;
    }
    
    // sampling every max_int_frames
    int_frame_cnt++;
    if(int_frame_cnt < max_int_frames){
        return false;
    }else{
        int_frame_cnt = 0;
    }
    
    // learn background model
    Mat fg_mask;
    MOG2(context->bgr, fg_mask, -1);
    learning_frame++;
    
    context->depth.copyTo(depth_bg);
    
    return learning_frame >= max_learning_frames;
}

void MaskGenerator::start_learning(){
    learning_frame = 0;
    MOG2 = BackgroundSubtractorMOG2();
    max_learning_frames = 300;
}

/*
    generate hand mask
*/

void MaskGenerator::update(AppContext *context, bool learn_bg){
 
    //Mat fg_mask = context->depth - depth_bg;
    //cv::threshold(fg_mask, fg_mask, 2, 255, CV_THRESH_BINARY);

    // background subtraction
    Mat fg_mask;
    Mat bgr;
    GaussianBlur(context->bgr, bgr, cv::Size(7,7), 1,1);
    MOG2(bgr, fg_mask, learn_bg ? -1 : 0);
    
    
    
    fg_mask.copyTo(context->raw_mask);
    
    // threshold
    fg_mask = fg_mask > 240;
    
    // smooth
    medianBlur(fg_mask, fg_mask, 5);
    
    // erosion
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                                        cv::Size( 2*erosion_size+1, 2*erosion_size+1 ),
                                        cv::Point( erosion_size, erosion_size ) );
	erode( fg_mask, fg_mask, element );
    
    element = getStructuringElement( MORPH_ELLIPSE,
                                    cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                    cv::Point( dilation_size, dilation_size ) );
    
    dilate(fg_mask, fg_mask, element);
    
    
    //more smoothing
	medianBlur(fg_mask, fg_mask, 3);
	GaussianBlur(fg_mask, fg_mask, cv::Size(3,3), 5);
	fg_mask = fg_mask > 240;

    // copy
    fg_mask.copyTo(context->mask);
}