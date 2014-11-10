//
//  train_async.h
//  HandDemo
//
//  Created by Akanksha Saran on 10/28/14.
//
//

#ifndef HandDemo_train_async_h
#define HandDemo_train_async_h

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"

#include "ofMain.h"
#include "ofThread.h"

class TrainAsync: public ofThread{

private:
    string basename;
    string img_prefix;
    string msk_prefix;
    string model_prefix;
    string globfeat_prefix;
    string feature_set;
    int step_size;
    
    bool busy;
    
    Mat _color_img;
    Mat _mask_img;
    int _k;
    
    
    void train_async(Mat &color_img, Mat &mask_img, int k){
        cout << "HandDetector::trainModels()" << endl;
        
        stringstream ss;
        
        
        LcFeatureExtractor	_extractor;
        LcRandomTreesR		_classifier;
        
        
        _extractor.set_extractor(feature_set);
        
        
        //////////////////////////////////////////
        //										//
        //		 EXTRACT/SAVE HISTOGRAM			//
        //										//
        //////////////////////////////////////////
        
        Mat globfeat;
        computeColorHist_HSV(color_img, globfeat);
        
        ss.str("");
        ss << globfeat_prefix << "/hsv_" + basename + "_" << k << ".xml";
        //ss << globfeat_prefix << "hsv_histogram_" << k << ".xml";
        cout << "  Writing global feature: " << ss.str() << endl;
        
        FileStorage fs;
        fs.open(ss.str(),FileStorage::WRITE);
        fs << "globfeat" << globfeat;
        fs.release();
        
        
        //////////////////////////////////////////
        //										//
        //		  TRAIN/SAVE CLASSIFIER			//
        //										//
        //////////////////////////////////////////
        
        Mat desc;
        Mat lab;
        vector<KeyPoint> kp;
        
        mask_img.convertTo(mask_img,CV_8UC1);
        _extractor.work(color_img, desc, mask_img, lab, step_size, &kp);
        
        
        _classifier.train(desc,lab);
        
        ss.str("");
        ss << model_prefix << "/model_" + basename + "_"+ feature_set + "_" << k;
        _classifier.save(ss.str());
        
        _classifier.release();
    }
    
    
    void computeColorHist_HSV(Mat &src, Mat &hist)
    {
        
        int bins[] = {4,4,4};
        if(src.channels()!=3) exit(1);
        
        //Mat tmp;
        //src.copyTo(tmp);
        
        Mat hsv;
        cvtColor(src,hsv,CV_BGR2HSV_FULL);
        
        int histSize[] = {bins[0], bins[1], bins[2]};
        Mat his;
        his.create(3, histSize, CV_32F);
        his = Scalar(0);
        CV_Assert(hsv.type() == CV_8UC3);
        MatConstIterator_<Vec3b> it = hsv.begin<Vec3b>();
        MatConstIterator_<Vec3b> it_end = hsv.end<Vec3b>();
        for( ; it != it_end; ++it )
        {
            const Vec3b& pix = *it;
            his.at<float>(pix[0]*bins[0]/256, pix[1]*bins[1]/256,pix[2]*bins[2]/256) += 1.f;
        }
        
        // ==== Remove small values ==== //
        float minProb = 0.01;
        minProb *= hsv.rows*hsv.cols;
        Mat plane;
        const Mat *_his = &his;
        
        NAryMatIterator itt = NAryMatIterator(&_his, &plane, 1);
        threshold(itt.planes[0], itt.planes[0], minProb, 0, THRESH_TOZERO);
        double s = sum(itt.planes[0])[0];
        
        // ==== Normalize (L1) ==== //
        s = 1./s * 255.;
        itt.planes[0] *= s;
        itt.planes[0].copyTo(hist);
        
        
    }
    
public:
    TrainAsync(){}
    
    void setup(string basename, string img_prefix,string msk_prefix, string model_prefix,string globfeat_prefix, string feature_set, int step_size){
        this->basename = basename;
        this->img_prefix = img_prefix;
        this->msk_prefix = msk_prefix;
        this->model_prefix = model_prefix;
        this->globfeat_prefix = globfeat_prefix;
        this->feature_set = feature_set;
        this->step_size = step_size;
        busy = false;
    }
    
    
    bool is_busy(){
        return busy;
    }
    
    void train(cv::Mat &color_img, cv::Mat &mask_img, int k){
        color_img.copyTo(_color_img);
        mask_img.copyTo(_mask_img);
        _k = k;

        busy = true;
        startThread();
    }
    
    
    void threadedFunction(){
        cv::Mat color_img;
        cv::Mat mask_img;
        int k;
        
        if(lock()){
            _color_img.copyTo(color_img);
            _mask_img.copyTo(mask_img);
            k = _k;
            unlock();
            
            cout << "============================= training " << k << " =============================" << endl;
            
            train_async(color_img, mask_img, k);
            
        }else{
            cout << "unable to lock in train_async.h" << endl;
        }
        
        busy = false;
    }
};


#endif
