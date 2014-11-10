//
//  endEffector.h
//  QoLT
//
//  Created by Akanksha Saran on 10/21/14.
//
//

#ifndef __QoLT__endEffector__
#define __QoLT__endEffector__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include <cstring>

#include <opencv2/flann/config.h>
#include <algorithm>
#include <opencv2/legacy/legacy.hpp>		// EM
#include <opencv2/contrib/contrib.hpp>		// colormap
#include <opencv2/nonfree/nonfree.hpp>		// SIFT

//#include "Classifier.h"
//#include "LcBasic.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/ml/ml.hpp"
#include <stdlib.h>
#include <iomanip>

#include <cmath>
#include <vector>


using namespace cv;
using namespace std;

class endEffector
{
public:
        
    CvRTrees		_classifier1;                    //grasp classifier trained
    CvRTrees		_classifier2; 
    vector<double>              _moments;
    vector<double>              _haar;
    //vector<float>  		label_distr;
    Mat				_bin;				// hand mask - ppr
    Mat             _patch;
    Mat Features;

    int grasp_label;
    Mat patchGray;
    
    void initialize(const char* xml_1, const char* xml_2);
    double count_whitePixels(cv::Mat &patch);
    void getPatch(Mat &bin);
    vector<double> calc_moments(Mat &patch);
    vector<double> calc_haar(Mat &patch);
    void computeFeatures();
    //void trainInit(Mat &TrFeatures1, Mat &TrFeatures2, Mat &TrLabels1, Mat &TrLabels2);
    //void train(Mat &feature, Mat &label, string classifier_name);
    int test();
    
    float label_distr[2];
    
private:
    
};


#endif /* defined(__QoLT__endEffector__) */
