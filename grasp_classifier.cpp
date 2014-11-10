//
//  grasp_classifier.cpp
//  HandDemo
//
//  Created by Akanksha Saran on 10/26/14.
//
//

#include "grasp_classifier.h"

void GraspClassifier::setup(AppContext *context){
    
    string xml_1 = context->data_root + "/medium_wrap.xml";
    string xml_2 = context->data_root + "/precision_disc.xml";
    
    _grasp.initialize(xml_1.c_str(), xml_2.c_str());
}

void GraspClassifier::update(AppContext *context){
    _grasp.getPatch(context->hand_mask);
    _grasp.computeFeatures();
    
    context->grasp_label = _grasp.test();
    
    context->grasp_scores[0] = _grasp.label_distr[0];
    context->grasp_scores[1] = _grasp.label_distr[1];
    
}