//
//  object_detector.cpp
//  HandDemo
//
//  Created by Akanksha Saran on 10/26/14.
//
//

#include "object_detector.h"

void ObjectDetector::setup(AppContext *context){
    _threshold = 85;
    _artk.setup(context->width, context->height);
    _artk.setThreshold(_threshold);
    
    context->artk = &_artk;
}

void ObjectDetector::update(AppContext *context){
    
    cv::Mat mat;
    cv::cvtColor(context->bgr, mat, CV_BGR2GRAY);
    
    _artk.update(mat.data);
}

void ObjectDetector::draw(ofRectangle &viewport){
    //_artk.draw(viewport.x, viewport.y, viewport.width, viewport.height);
}