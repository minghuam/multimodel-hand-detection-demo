#ifndef HandDemo_torso_com_h
#define HandDemo_torso_com_h

#include "app_context.h"
#include "opencv2/opencv.hpp"

using namespace cv;

class TorsoCom{
private:
    int _min_head_pix_cnt;
    int _head_depth_range;
public:
    void setup(AppContext *context);
    void update(AppContext *context);
};

/*
class TorsoCom{
private:
    float COM_x;
    float COM_y;
    float COM_z;
    int bins;
    //Mat rgb;		     			 //CV_8UC3 - color
    Mat depth;               			 //CV_8UC1 - grayscale
    Mat threshDepth;			         //CV_8U - grayscale
    int nearThreshold;
    int farThreshold;
    unsigned char depth_hist[32];
    
    void initialize(int no_of_bins);
    void threshold_depth();
    void computeCOM();
    void computeCOM2();
    
public:
    void setup(AppContext *context) {
        initialize(32);
    }
    
    void update(AppContext *context) {
        //
        context->depth.copyTo(depth);
        //threshold_depth();
        //computeCOM();
        computeCOM2();
        
        context->center_of_mass[0] = COM_x;
        context->center_of_mass[1] = COM_y;
        context->center_of_mass[2] = COM_z;
    }
};
*/

#endif
