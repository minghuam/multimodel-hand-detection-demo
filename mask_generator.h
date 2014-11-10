//
//  mask_generator.h
//  HandDemo
//
//  Created by Minghuang Ma on 10/22/14.
//
//

#ifndef HandDemo_mask_generator_h
#define HandDemo_mask_generator_h

#include "app_context.h"
#include "opencv2/opencv.hpp"
#include "opencv2/video/background_segm.hpp"

using namespace cv;

class MaskGenerator{
private:
	BackgroundSubtractorMOG2 MOG2;
	int erosion_size;
    int dilation_size;
    
    int learning_frame;
    int max_learning_frames;
    
    Mat depth_bg;
    
public:
    void setup(AppContext *context);
    void start_learning();
    bool learn(AppContext *context);
    void update(AppContext *context, bool learn_bg = false);
};


#endif
