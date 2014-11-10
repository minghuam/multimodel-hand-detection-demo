#ifndef Define_PrepData
#define Define_PrepData

//created by Akanksha Saran  10/27/2014

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/config.h>
#include <opencv2/legacy/legacy.hpp>		// EM
#include <opencv2/contrib/contrib.hpp>		// colormap
#include <opencv2/nonfree/nonfree.hpp>		// SIFT

#include <algorithm>
#include "FeatureComputer.hpp"

using namespace std;
using namespace cv;

class PrepData{
public:
	//color imgs    CV_8UC3
	Mat obj_img;
	Mat hand_img;
	Mat obj_mask;
	Mat hand_mask;

	Mat obj_label;
	Mat hand_label;

	Mat obj_desc;
	Mat hand_desc;


	void init(Mat &O, Mat &H, Mat &Om, Mat & Hm);
	void generate_features(string feature_set);

	

};

#endif
