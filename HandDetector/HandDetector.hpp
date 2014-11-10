/*
 *  HandDetector.h
 *  TrainHandModels
 *
 *  Created by Kris Kitani on 4/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef HANDDETECTOR_H
#define HANDDETECTOR_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"

using namespace cv;
using namespace std;

class HandDetector
{
public:
	
	void loadMaskFilenames(vector<string> &filenames, string msk_prefix);
	vector<string> _filenames;
	
    //void trainModels(Mat &color_img, string basename, string model_prefix,string globfeat_prefix, string feature_set, Mat &lab, Mat &desc, int k );
    
    void trainModels(string basename, string img_prefix,string msk_prefix,string model_prefix,string feat_prefix, string feature_set,Mat &color_img, Mat &mask_img, int i);
    void trainModels(string basename, string img_prefix,string msk_prefix, string model_prefix,string globfeat_prefix, string feature_set, int k, Mat &color_img, Mat &mask_img, int step_size);
    
    void trainModels(string basename, string img_prefix,string msk_prefix,string model_prefix,string feat_prefix, string feature_set);
	
    string _feature_set;
	
	void testInitialize(string model_prefix,string feat_prefix, int num_models, string feature_set, int knn);
	
	vector<LcRandomTreesR>		_classifier;
	vector<int>					_indices; 
	vector<float>				_dists; 
	int							_knn;
	flann::Index				_searchtree;
	flann::IndexParams			_indexParams;
	LcFeatureExtractor			_extractor;
	Mat							_hist_all;				// do not destroy!
	

	void test(Mat &img, float color_code = 0.85f);
	void test(Mat &img,Mat &dsp, float color_code = 0.85f);
	void test(Mat &img,int num_models, float color_code = 0.85f);
	void test(Mat &img,Mat &dsp,int num_models, float color_code = 0.85f);
	
	Mat							_descriptors;
	Mat							_response_vec;
	Mat							_response_img;
	vector<KeyPoint>			_kp;
	Mat							_dsp;
	cv::Size					_sz;
	int							_bs;
	Mat							_response_avg;
	
	Mat							_raw;				// raw response
	Mat							_blu;				// blurred image
	Mat							_ppr;				// post processed
    
    LcFeatureExtractor	_train_extractor;
    LcRandomTreesR		_train_classifier;
    
	
	Mat postprocess(Mat &img,vector<Point2f> &pt);
	
	void computeColorHist_HSV(Mat &src, Mat &hist);
	void colormap(Mat &src, Mat &dst, int do_norm, float color_code = 0.85f);
	void rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, cv::Size s, int bs);
	
private:
	
};

#endif