//
//  object_detector.h
//  HandDemo
//
//  Created by Minghuang Ma on 10/22/14.
//
//

#ifndef HandDemo_object_detector_h
#define HandDemo_object_detector_h

#include "app_context.h"
#include "opencv2/opencv.hpp"

#include "ofxARToolkitPlus.h"

class ObjectDetector{
private:
    int _threshold;
    ofxARToolkitPlus _artk;
    
public:
    void setup(AppContext *context);
    void update(AppContext *context);
    void draw(ofRectangle &viewport);
};


#endif
