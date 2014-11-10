//
//  grasp_classifier.h
//  HandDemo
//
//  Created by Minghuang Ma on 10/22/14.
//
//

#ifndef HandDemo_grasp_classifier_h
#define HandDemo_grasp_classifier_h

#include "app_context.h"
#include "opencv2/opencv.hpp"
#include "endEffector.h"

class GraspClassifier{
private:
    endEffector _grasp;
    
public:
    
    void setup(AppContext *context);
    void update(AppContext *context);
    
};


#endif
