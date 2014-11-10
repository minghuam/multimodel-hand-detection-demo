/*
 *  PrepData.cpp
 *  
 *
 *  Created by Akanksha Saran on 27/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "PrepData.h"


void PrepData::init(Mat &O, Mat &H, Mat &Om, Mat & Hm)
{
	obj_img = O.clone();
	hand_img = H.clone();
	obj_mask = Om.clone();
	hand_mask = Hm.clone();
}

void PrepData::generate_features(string feature_set)
{
	
	
	vector<KeyPoint> kp1;
	vector<KeyPoint> kp2;
	
	Mat obj_mask1; Mat hand_mask1;

	LcFeatureExtractor extractor_obj;
	LcFeatureExtractor extractor_hand;
	
	extractor_obj.set_extractor(feature_set);
	extractor_hand.set_extractor(feature_set);
	
	obj_mask.convertTo(obj_mask1,CV_8UC1);
	hand_mask.convertTo(hand_mask1,CV_8UC1);
    
    Mat obj_posFeat;
    Mat obj_negFeat;
    Mat hand_posFeat;
    Mat hand_negFeat;

    Mat desc_obj;
    Mat lab_obj;
    
    Mat desc_hand;
    Mat lab_hand;
    
	extractor_obj.work(obj_img, desc_obj, obj_mask1, lab_obj, 1, &kp1);
	extractor_hand.work(hand_img, desc_hand, hand_mask1, lab_hand, 1, &kp2);
    
    
    //cout<<"$$$$"<<lab_hand.size()<<"$$$$$"<<endl;

	Mat pos_feat_obj = Mat::zeros(desc_obj.rows,desc_obj.cols, CV_32F) ;
	Mat neg_feat_obj = Mat::zeros(desc_obj.rows,desc_obj.cols, CV_32F) ;
	Mat pos_feat_hand = Mat::zeros(desc_hand.rows,desc_hand.cols, CV_32F) ;
	Mat neg_feat_hand = Mat::zeros(desc_hand.rows,desc_hand.cols, CV_32F);

	int pos_cnt_obj = 0;
	int neg_cnt_obj = 0;


	//go through desc_hand and desc_obj and corresponding labels to keep all positive, sample half negative
	for(int i=0; i<lab_obj.rows; i++)
	{
		if(lab_obj.at<float>(i,0)==1)
		{
			for(int j=0; j<desc_obj.cols; j++)
			{
				pos_feat_obj.at<float>(pos_cnt_obj, j) = desc_obj.at<float>(i,j);
			}
			pos_cnt_obj +=1;
		}

		if(lab_obj.at<float>(i,0)==0)
		{
			for(int j=0; j<desc_obj.cols; j++)
			{
				neg_feat_obj.at<float>(neg_cnt_obj, j) = desc_obj.at<float>(i,j);
			}
			neg_cnt_obj +=1;
		}

	}

	int pos_cnt_hand = 0;
	int neg_cnt_hand = 0;

	for(int i=0; i<lab_hand.rows; i++)
	{
        
        if(lab_hand.at<float>(i,0)==1)
		{
            cout<<"yes:"<<lab_hand.at<float>(i,0)<<endl;

			for(int j=0; j<desc_hand.cols; j++)
			{
				pos_feat_hand.at<float>(pos_cnt_hand, j) = desc_hand.at<float>(i,j);
			}
			pos_cnt_hand +=1;
		}

		if(lab_obj.at<float>(i,0)==0)
		{
			for(int j=0; j<desc_hand.cols; j++)
			{
				neg_feat_hand.at<float>(neg_cnt_hand, j) = desc_hand.at<float>(i,j);
			}
			neg_cnt_hand +=1;
		}

	}

    cout<<"$$$$"<<pos_feat_hand.size()<<"$$$$$"<<endl;
    cout<<"$$$$"<<neg_feat_hand.size()<<"$$$$$"<<endl;
    
	int no_of_feats1 = min(pos_cnt_obj, neg_cnt_obj);
	int no_of_feats2 = min(pos_cnt_hand, neg_cnt_hand);

	int factor1 = floor(max(pos_cnt_obj, neg_cnt_obj)/min(pos_cnt_obj, neg_cnt_obj));
	int factor2 = floor(max(pos_cnt_hand, neg_cnt_hand)/min(pos_cnt_hand, neg_cnt_hand));

	int l1 = (pos_cnt_obj> neg_cnt_obj ? 1:-1) ;
	int l2 = (pos_cnt_hand> neg_cnt_hand ? 1:-1) ;

	obj_posFeat = Mat::zeros(no_of_feats1, desc_obj.cols, CV_32F);
	obj_negFeat = Mat::zeros(no_of_feats1, desc_obj.cols, CV_32F);
	hand_posFeat = Mat::zeros(no_of_feats2, desc_hand.cols, CV_32F);
	hand_negFeat = Mat::zeros(no_of_feats2, desc_hand.cols, CV_32F);

    obj_label = Mat::zeros(2*no_of_feats1, 1, 5);
	hand_label = Mat::zeros(2*no_of_feats2, 1, 5);

	for(int i=0; i<obj_posFeat.rows; i++)
	{	
		for(int j=0; j<obj_posFeat.cols; j++)
		{
			if(l1==1 && (i%factor1)==0)
			{	
				obj_posFeat.at<float>(i,j) = pos_feat_obj.at<float>(i,j);
			}
			else
			{
				obj_posFeat.at<float>(i,j) = pos_feat_obj.at<float>(i,j);
			}
			
		}
		obj_label.at<float>(i,0) = 1;
	}


	for(int i=0; i<obj_negFeat.rows; i++)
	{	
		for(int j=0; j<obj_negFeat.cols; j++)
		{
			if(l1==-1 && (i%factor1)==0)
			{	
				obj_negFeat.at<float>(i,j) = neg_feat_obj.at<float>(i,j);
			}
			else
			{
				obj_negFeat.at<float>(i,j) = neg_feat_obj.at<float>(i,j);
			}
		}
		obj_label.at<float>(i,0) = 0;
	}



	for(int i=0; i<hand_posFeat.rows; i++)
	{	
		for(int j=0; j<hand_posFeat.cols; j++)
		{
			if(l1==1 && (i%factor1)==0)
			{	
				hand_posFeat.at<float>(i,j) = pos_feat_hand.at<float>(i,j);
			}
			else
			{
				hand_posFeat.at<float>(i,j) = pos_feat_hand.at<float>(i,j);
			}
		}
		hand_label.at<float>(i,0) = 1;
	}


	for(int i=0; i<hand_negFeat.rows; i++)
	{	
		for(int j=0; j<hand_negFeat.cols; j++)
		{
			if(l1==-1 && (i%factor1)==0)
			{	
				hand_negFeat.at<float>(i,j) = neg_feat_hand.at<float>(i,j);
			}
			else
			{
				hand_negFeat.at<float>(i,j) = neg_feat_hand.at<float>(i,j);
			}
		}
		hand_label.at<float>(i,0) = 0;	
	}

	

	obj_desc.push_back(obj_posFeat);
	obj_desc.push_back(obj_negFeat);
	hand_desc.push_back(hand_posFeat);
	hand_desc.push_back(hand_negFeat);
    
    cout<<"****"<<obj_desc.size()<<"******"<<obj_desc.type()<<endl;
    cout<<"****"<<hand_desc.size()<<"******"<<hand_desc.type()<<endl;

}


