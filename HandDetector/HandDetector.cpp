/*
 *  HandDetector.cpp
 *  TrainHandModels
 *
 *  Created by Kris Kitani on 4/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HandDetector.hpp"

void HandDetector::loadMaskFilenames(vector<string> &filenames, string msk_prefix)
{
    // added by minghuam
    filenames.clear();
    
	string cmd = "ls " + msk_prefix + " > maskfilename.txt";
    cout<<cmd.c_str()<<endl;
	system(cmd.c_str());
	
	ifstream fs;
	fs.open("maskfilename.txt");
	string val;
	while(fs>>val) filenames.push_back(val);
    fs.close();
}


/*void HandDetector::trainModels(Mat &color_img, string basename, string model_prefix,string globfeat_prefix, string feature_set, Mat &lab, Mat &desc, int k )
{
    cout << "HandDetector::trainModels()" << endl;
    
    LcFeatureExtractor	_extractor;
    LcRandomTreesR		_classifier;
    
    stringstream ss;
    
    //////////////////////////////////////////
    //										//
    //		 EXTRACT/SAVE HISTOGRAM			//
    //										//
    //////////////////////////////////////////
    
    Mat globfeat;
    computeColorHist_HSV(color_img,globfeat);
    
    ss.str("");
    ss << globfeat_prefix << "hsv_histogram_" << k << ".xml";
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
    
    
    _classifier.train(desc,lab);
    
    //cout<<"desc matrix type: "<< desc.type()<<endl;
    
    ss.str("");
    ss << model_prefix << "model_" + basename + "_"+ feature_set + "_" << k;
    _classifier.save(ss.str());
    
    //k++;
    
    cout << k << endl;
    
    //}
    
}*/


void HandDetector::trainModels(string basename, string img_prefix,string msk_prefix,string model_prefix,string feat_prefix, string feature_set, Mat &color_img, Mat &mask_img, int i)
{
    
	stringstream ss;
	
	_feature_set = feature_set;
	_extractor.set_extractor(feature_set);
	
			
		//////////////////////////////////////////
		//										//
		//		 EXTRACT/SAVE HISTOGRAM			//
		//										//
		//////////////////////////////////////////
		
		Mat hist;
		computeColorHist_HSV(color_img,hist);
		
		ss.str("");
		ss << feat_prefix << "/hsv_" + basename + "_" << i << ".xml";
		cout << ss.str() << endl;
		
		FileStorage fs;
		fs.open(ss.str(),FileStorage::WRITE);
		fs << "globfeat" << hist;
		fs.release();
		
		//hist_all.push_back(hist);
		
		//////////////////////////////////////////
		//										//
		//		  TRAIN/SAVE CLASSIFIER			//
		//										//
		//////////////////////////////////////////
		
		Mat desc;
		Mat lab;
		vector<KeyPoint> kp;

		mask_img.convertTo(mask_img,CV_8UC1);
		_train_extractor.work(color_img, desc, mask_img, lab,1, &kp);
		_train_classifier.train(desc,lab);
		
		ss.str("");
		ss << model_prefix << "/model_" + basename +"_"+ feature_set + "_" << i;
		cout << ss.str() << endl;
		_train_classifier.save(ss.str());
}

 void HandDetector::trainModels(string basename, string img_prefix,string msk_prefix,string model_prefix,string feat_prefix, string feature_set)
{
    
    stringstream ss;
    
    LcFeatureExtractor	_extractor;
    LcRandomTreesR		_classifier;
    
    _feature_set = feature_set;
    _extractor.set_extractor(feature_set);
    
    loadMaskFilenames(_filenames, msk_prefix);
    
    for(int i=0;i<(int)_filenames.size();i++)
    {
        cout << _filenames[i] << endl;
        string f_str = _filenames[i].substr(0,8);
        //cout<<f_str<<endl;
        
        ss.str("");
        ss << img_prefix + f_str + ".jpg";
        cout << ss.str() << endl;
        Mat color_img = imread(ss.str());				// Loaded as BGR by default
        
        // commented by minghuam
        //cvtColor(color_img,color_img,CV_BGR2RGB);		// Convert to RGB
        
        ss.str("");
        ss << msk_prefix << _filenames[i];
        cout << ss.str() << endl;
        Mat mask_img = imread(ss.str(),0);
        
        imshow("color",color_img);
        imshow("mask",mask_img);
        waitKey(1);
        
        //////////////////////////////////////////
        //										//
        //		 EXTRACT/SAVE HISTOGRAM			//
        //										//
        //////////////////////////////////////////
        
        Mat hist;
        computeColorHist_HSV(color_img,hist);
        
        ss.str("");
        ss << feat_prefix << "/hsv_" + basename + "_" << i << ".xml";
        cout << ss.str() << endl;
        
        FileStorage fs;
        fs.open(ss.str(),FileStorage::WRITE);
        fs << "globfeat" << hist;
        fs.release();
        
        //hist_all.push_back(hist);
        
        //////////////////////////////////////////
        //										//
        //		  TRAIN/SAVE CLASSIFIER			//
        //										//
        //////////////////////////////////////////
        
        Mat desc;
        Mat lab;
        vector<KeyPoint> kp;
        
        mask_img.convertTo(mask_img,CV_8UC1);
        _extractor.work(color_img, desc, mask_img, lab,1, &kp);
        _classifier.train(desc,lab);
        
        ss.str("");
        ss << model_prefix << "/model_" + basename +"_"+ feature_set + "_" << i;
        cout << ss.str() << endl;
        _classifier.save(ss.str());
        
    }
    
}

void HandDetector::trainModels(string basename, string img_prefix,string msk_prefix, string model_prefix,string globfeat_prefix, string feature_set, int k, Mat &color_img, Mat &mask_img, int step_size){
    
    cout << "HandDetector::trainModels()" << endl;
    
    //imshow("color",color_img);
    //imshow("mask",mask_img);
    //waitKey(1);
    
    stringstream ss;
    
    
    LcFeatureExtractor	_extractor;
    LcRandomTreesR		_classifier;
    
    
    _feature_set = feature_set;
    _extractor.set_extractor(feature_set);
    
    
    //////////////////////////////////////////
    //										//
    //		 EXTRACT/SAVE HISTOGRAM			//
    //										//
    //////////////////////////////////////////
    
    Mat globfeat;
    computeColorHist_HSV(color_img,globfeat);
    
    ss.str("");
    //ss << globfeat_prefix << "hsv_histogram_" << k << ".xml";
    ss << globfeat_prefix << "/hsv_" + basename + "_" << k << ".xml";
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
    
    //mask_img.convertTo(mask_img,CV_8UC1);
    
    _extractor.work(color_img, desc, mask_img, lab, step_size, &kp);
    
    //cv::imshow("color", color_img);
    //cv::imshow("maks", mask_img);
    
    //cout << mask_img.type() << endl;
    //cout << cv::countNonZero(lab) << endl;
    
    _classifier.train(desc,lab);
    
    ss.str("");
    ss << model_prefix << "/model_" + basename + "_"+ feature_set + "_" << k;
    _classifier.save(ss.str());
    
    _classifier.release();
    
}


void HandDetector::testInitialize(string model_prefix,string feat_prefix, int num_models, string feature_set, int knn)
{
    
    // added by minghuam
    cv::Mat empty;
    _hist_all = empty;
	
	stringstream ss;
	
	//////////////////////////////////////////
	//										//
	//		     FEATURE EXTRACTOR			//
	//										//
	//////////////////////////////////////////
	
	cout << "set extractor" << endl;
	_feature_set = feature_set;
	_extractor.set_extractor(_feature_set);
	
	
	//////////////////////////////////////////
	//										//
	//		       LOAD CLASSIFIERS			//
	//										//
	//////////////////////////////////////////
	
	{
		string cmd;
		cmd = "find " + model_prefix + " -name *.xml -print > modelfilename.txt";
		cout << cmd << endl;
		system(cmd.c_str());
		
        cout<<"here"<<endl;
        
		ifstream fs;
		vector<string> filenames;
		fs.open("modelfilename.txt");
		filenames.clear();
		
		string val;
		while(fs>>val) filenames.push_back(val);
        fs.close();
		
		num_models = (int)filenames.size();
		
		cout << "Load class" << endl;
		_classifier = vector<LcRandomTreesR>(num_models);
		
		for(int i=0;i<num_models;i++)
		{
            //cout<<filenames[i]<<"**********"<<endl;
			_classifier[i].load_full(filenames[i]);
		}
	}
	
	
	//////////////////////////////////////////
	//										//
	//		       LOAD HISTOGRAM			//
	//										//
	//////////////////////////////////////////
	
	{
        

        
		string cmd;
		cmd = "find " + feat_prefix + " -name *.xml -print > histfilename.txt";
		cout << cmd << endl;
		system(cmd.c_str());
		
		ifstream fs;
		vector<string> filenames;
		fs.open("histfilename.txt");
		filenames.clear();
		
		string val;
		while(fs>>val) filenames.push_back(val);
        
        fs.close();
		
		num_models = (int)filenames.size();
		cout<<"HELLO"<<endl;
        cout<<filenames[0]<<endl;
        for(int i=0;i<num_models;i++)
		{
            //cout<<num_models<<endl;
			Mat hist;
			
			////ss.str("");
			////ss << feat_prefix << "hsv_" << i << ".xml";
			////cout << ss.str() << endl;
            
			//cout << filenames[i] << endl;
            
            FileStorage fs;
			fs.open(filenames[i],FileStorage::READ);
			////fs.open(ss.str(),FileStorage::READ);
            
			fs["globfeat"] >> hist;
            
			fs.release();
			
			_hist_all.push_back(hist);
            
			
		}
	}
	
	if(_hist_all.rows != (int)_classifier.size())
    {
        cout << "ERROR: Mismatch.\n";
    }
	
	//////////////////////////////////////////
	//										//
	//		       KNN CLASSIFIER			//
	//										//
	//////////////////////////////////////////
	
	cout << "Building FLANN search structure...";
	//_indexParams = new flann::LinearIndexParams;
	//_indexParams = *new flann::KDTreeIndexParams;
	_indexParams = *new flann::KMeansIndexParams;
	
	//_indexParams = *new flann::AutotunedIndexParams(0.9, 0.0, 0.0, 1.0);
	
	_searchtree  = *new flann::Index(_hist_all, _indexParams);
	
	_knn		= knn;						//number of nearest neighbors 
	_indices	= vector<int> (_knn); 
	_dists		= vector<float> (_knn); 
		

}

void HandDetector::test(Mat &img, float color_code)
{
	Mat tmp = Mat();
	test(img,tmp,1, color_code);
}

void HandDetector::test(Mat &img, Mat &dsp, float color_code)
{
	test(img,dsp,1, color_code);
}

void HandDetector::test(Mat &img, int num_models, float color_code)
{
	Mat tmp = Mat();
	test(img,tmp,num_models, color_code);
}


void HandDetector::test(Mat &img, Mat &dsp, int num_models, float color_code)
{
	if(num_models>_knn) return;
	
	Mat hist;
	computeColorHist_HSV(img,hist);									// extract hist
	
	//_searchtree.knnSearch(hist,
	//					   _indices, _dists, 
	//					   _knn, flann::SearchParams(4));			// probe search
    
	_searchtree.knnSearch(hist, _indices, _dists, _knn);			// probe search
	
	_extractor.work(img,_descriptors,3,&_kp);
	
	if(!_response_avg.data) _response_avg = Mat::zeros(_descriptors.rows,1,CV_32FC1); 
	else _response_avg *= 0;
	
	float norm = 0;
	for(int i=0;i<num_models;i++)
	{
		int idx = _indices[i];
		_classifier[idx].predict(_descriptors,_response_vec);		// run classifier
		
		_response_avg += _response_vec*float(pow(0.9f,(float)i));
		norm += float(pow(0.9f,(float)i));
	}
	
	_response_avg /= norm;
	
	_sz = img.size();
	_bs = _extractor.bound_setting;	
	rasterizeResVec(_response_img,_response_avg,_kp,_sz,_bs);		// class one
	
	colormap(_response_img,_raw,1);
	
	//_raw = _dsp.clone();
	
	//Mat pp;
	vector<Point2f> pt;
	_ppr = postprocess(_response_img,pt);
    
	colormap(_ppr,_ppr,1, color_code);
	//cvtColor(_ppr,_ppr,CV_GRAY2BGR);
    
    
	if(0)
	{
		
		imshow("hd response",_dsp);
		imshow("hd img",img);	
		imshow("_ppr",_ppr);	
		waitKey(1);
	}
	
	//dsp = pp;
	if(!dsp.data) dsp = _dsp;					// pass by reference?
	
}


Mat HandDetector::postprocess(Mat &img,vector<Point2f> &pt)
{
	Mat tmp;
	//GaussianBlur(img,tmp,cv::Size(31,31),0,0,BORDER_REFLECT);
    GaussianBlur(img,tmp,cv::Size(15,15),0,0,BORDER_REFLECT);
    
	//Mat dsp;
	colormap(tmp,_blu,1);
	//imshow("dsp",dsp);
	
	tmp = tmp > 0.04;
	
	vector<vector<cv::Point> > co;
	vector<Vec4i> hi;
	
	//findContours(tmp,co,hi,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    findContours(tmp,co,hi,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
	tmp *= 0;
	
	//Moments m;
	//vector<Point2f> pt;
	//for(int i=0;i<(int)co.size();i++)
	//{
	//	if(contourArea(Mat(co[i])) < 300) continue;
	//	drawContours(tmp, co,i, CV_RGB(255,255,255), CV_FILLED, CV_AA);
	//	m = moments(Mat(co[i]));
	//	pt.push_back(Point2f(m.m10/m.m00,m.m01/m.m00));
	//}
    
    float max_size = 0;
    for(int i=0;i<(int)co.size();i++)
    {
        float area = contourArea(Mat(co[i]));
        if(area < 300) continue;
        if( area > max_size)
        {
            tmp *= 0;
            drawContours(tmp, co,i, CV_RGB(255,255,255), CV_FILLED, CV_AA);
            max_size = contourArea(Mat(co[i]));
        }
    }
	
	return tmp;

}


void HandDetector::rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, cv::Size s, int bs)
{	
    if((img.rows!=s.height) || (img.cols!=s.width) || (img.type()!=CV_32FC1) ) img = Mat::zeros( s, CV_32FC1);
	
	for(int i = 0;i< (int)keypts.size();i++)
    {
		int r = floor(keypts[i].pt.y);
		int c = floor(keypts[i].pt.x);
		img.at<float>(r,c) = res.at<float>(i,0);
	}
}


void HandDetector::colormap(Mat &src, Mat &dst, int do_norm, float color_code)
{
	
	double minVal,maxVal;
	minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	
	//cout << "colormap minmax: " << minVal << " " << maxVal << " Type:" <<  src.type() << endl;
	
	Mat im;
	src.copyTo(im);
	
	if(do_norm) im = (src-minVal)/(maxVal-minVal);		// normalization [0 to 1]
	
	Mat mask;	
	mask = Mat::ones(im.size(),CV_8UC1)*255.0;	
	
	compare(im,0.01,mask,CMP_GT);						// one color values greater than X	
	
	
	Mat U8;
	im.convertTo(U8,CV_8UC1,255,0);
	
	Mat I3[3],hsv;
	I3[0] = U8 * color_code;
	I3[1] = mask;
	I3[2] = mask;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);
	
	
}


void HandDetector::computeColorHist_HSV(Mat &src, Mat &hist)
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

