//
//  hand_detector.cpp
//  HandDemo
//
//  Created by Minghuang Ma on 10/22/14.
//
//

#include "hand_detector.h"


void HandDetectorWrapper::setup(AppContext *context){
    
    
    cout << context->data_root << endl;
    
    root = "";
    root = context->data_root + "/_CLAB/";
    //cout << root << endl;
	
    //basename = "demo_may";
    basename = "demo_oct";
	
    // hand
    img_prefix	= root + "img/"    + basename + "/";
	msk_prefix	= root + "mask/"   + basename + "/";
    
    model_prefix = root + "models/" + basename;
    feat_prefix  = root + "globfeat/"+ basename;
    
    // object
    obj_msk_prefix = root + "obj_msk/" + basename + "/";
    obj_img_prefix = root + "obj_img/" + basename + "/";
	
    obj_model_prefix = root + "obj_models/" + basename;
    obj_feat_prefix  = root + "obj_globfeat/"+ basename;
    
    
    feature_set = "rvl";
    knn = 10;
    num_models = 31;
    
    sample_cnt = 0;
    num_train_imgs = 31;
    
    save_cnt = 0;
    num_grasp_imgs = 1000;
    grasp_mask_prefix = root + "grasp/" + basename + "/mask/";
    grasp_img_prefix = root + "grasp/" + basename + "/img/";
    grasp_depth_prefix = root + "grasp/" + basename + "/depth/";
    
    hd_hand.loadMaskFilenames(hand_filenames, msk_prefix);
    hd_hand.testInitialize(model_prefix, feat_prefix, num_models, feature_set, knn);

    hd_obj.loadMaskFilenames(obj_filenames, obj_msk_prefix);
    hd_obj.testInitialize(obj_model_prefix, obj_feat_prefix, num_models, feature_set, knn);
    
    
    _trainAsync_hand.setup(basename, img_prefix, msk_prefix, model_prefix, feat_prefix, feature_set, 1);
    _trainAsync_obj.setup(basename, obj_img_prefix, obj_msk_prefix, obj_model_prefix, obj_feat_prefix, feature_set, 1);
    
}

void HandDetectorWrapper::start_sampling(){
    sample_cnt = 0;
}

/*
    collect hand images and masks for training models
*/
int HandDetectorWrapper::sample(AppContext *context){
    
    const int max_int_frames = 10;
    static int int_frame_cnt = 0;
    
    // sampling every max_int_frames
    int_frame_cnt++;
    if(int_frame_cnt < max_int_frames){
        if(sample_cnt == 0) return 1;
        else return sample_cnt;
    }else{
        int_frame_cnt = 0;
    }
    
    if(sample_cnt >= num_train_imgs){
        return 0;
    }
    
    char buf[128];
    sprintf(buf, "%08d.jpg", sample_cnt);
    string mask_file = buf;
    mask_file = msk_prefix + mask_file;
    
    sprintf(buf, "%08d.jpg", sample_cnt);
    string img_file = buf;
    img_file = img_prefix + img_file;
    
    sprintf(buf, "%08d.jpg", sample_cnt);
    string obj_msk_file = buf;
    obj_msk_file = obj_msk_prefix + obj_msk_file;
    
    sprintf(buf, "%08d.jpg", sample_cnt);
    string obj_img_file = buf;
    obj_img_file = obj_img_prefix + obj_img_file;
    
    
    cv::Mat mat = context->bgr(context->hand_roi);
    cv::imwrite(img_file, mat);
    
    mat = context->mask(context->hand_roi);
    cv::imwrite(mask_file, mat);
    
    mat = context->mask(context->obj_roi);
    cv::imwrite(obj_msk_file, mat);
    
    mat = context->bgr(context->obj_roi);
    cv::imwrite(obj_img_file, mat);
    
    sample_cnt++;
    
    return sample_cnt;
    //return sample_cnt > num_train_imgs;
}

/*
    train hand models
*/
int HandDetectorWrapper::train(AppContext *context){
    
    static int count = 0;
    
    if(_trainAsync_hand.is_busy()){
        // get back later
        return count;
    }
    
    if(count == num_models){
        hd_hand.testInitialize(model_prefix, feat_prefix, num_models, feature_set, knn);
        hd_obj.testInitialize(obj_model_prefix, obj_feat_prefix, num_models, feature_set, knn);
        count = 0;
        return count;
    }
    
    string hand_file = hand_filenames[count];
    string obj_file = obj_filenames[count];
    
    cv::Mat hand_img_ = cv::imread(img_prefix + hand_file);
    cv::Mat hand_msk_ = cv::imread(msk_prefix + hand_file);
    cv::Mat obj_img_ = cv::imread(obj_img_prefix + obj_file);
    cv::Mat obj_msk_ = cv::imread(obj_msk_prefix + obj_file);
    
    cv::cvtColor(hand_msk_, hand_msk_, CV_BGR2GRAY);
    cv::cvtColor(obj_msk_, obj_msk_, CV_BGR2GRAY);
    
    hand_img_.copyTo(context->current_training_img);
    hand_msk_.copyTo(context->current_training_msk);
    
    /*
    //start = clock();
    hd_obj.trainModels(basename, obj_img_prefix, obj_msk_prefix, obj_model_prefix, obj_feat_prefix, feature_set, count, obj_img_, obj_msk_, 1);
    //end = clock();
    //cout << "train one object image: " << (float)(end - start)/CLOCKS_PER_SEC;
    
    
    //clock_t start = clock();
    hd_hand.trainModels(basename, img_prefix, msk_prefix, model_prefix, feat_prefix, feature_set, count, hand_img_, hand_msk_, 1);
    //clock_t end = clock();
    //cout << "train one hand image: " << (float)(end - start)/CLOCKS_PER_SEC;
    */
    
    _trainAsync_hand.train(hand_img_, hand_msk_, count);
    _trainAsync_obj.train(obj_img_, obj_msk_, count);
    
    count++;
    
    return count;
}

bool HandDetectorWrapper::train_async(AppContext *context){
    
}

/*
    test
*/

void HandDetectorWrapper::update(AppContext *context){
    if(context->is_frame_new){
        
        // hand test
        cv::Mat img = context->bgr(context->hand_roi);
        hd_hand.test(img, knn, 0.85);
        
        
        // covert hand mask to gray mask
        cv::cvtColor(hd_hand._ppr, context->hand_mask, CV_BGR2GRAY);
        context->hand_mask = context->hand_mask > 0;
        
        // obj test
        img = context->bgr(context->hand_roi);
        hd_obj.test(img, knn, 0.45);
        
        // covert object mask to gray mask
        cv::cvtColor(hd_obj._ppr, context->obj_mask, CV_BGR2GRAY);
        context->obj_mask = context->obj_mask > 0;
        
        // fix hand mask
        cv::Mat common_region;
        cv::Mat temp;
        cv::bitwise_not(context->obj_mask, temp);
        cv::bitwise_and(context->hand_mask, temp, context->hand_mask);
        
        // prepare visulization image
        Mat color_mask = Mat::zeros(context->hand_roi.height, context->hand_roi.width, CV_8UC3);
        Mat chs[3];
        split(color_mask, chs);
        context->hand_mask.copyTo(chs[0]);
        context->obj_mask.copyTo(chs[1]);
        merge(chs, 3, color_mask);
        
        context->rgb.copyTo(context->hand_ppr);
        cv::addWeighted(context->hand_ppr(context->hand_roi), 0.7, color_mask, 0.3, 0, context->hand_ppr(context->hand_roi));
        
        
        //context->bgr(context->hand_roi).copyTo(img);
        //cv::Mat img2;
        //cv::addWeighted(img, 0.7, black, 0.3, 0, img2);
        //img2.copyTo(context->hand_ppr(context->hand_roi));
        
        /*
        // hand test
        cv::Mat img = context->bgr(context->hand_roi);
        hd_hand.test(img, knn, 0.85);
        
        cv::Mat img2;
        cv::addWeighted(img, 0.7, hd_hand._ppr, 0.3, 0, img2);
        context->rgb.copyTo(context->hand_ppr);
        cv::cvtColor(img2, img2, CV_BGR2RGB);
        img2.copyTo(context->hand_ppr(context->hand_roi));
        
        // covert hand mask to gray mask
        cv::cvtColor(hd_hand._ppr, context->hand_mask, CV_BGR2GRAY);
        context->hand_mask = context->hand_mask > 0;
        
        // obj test
        img = context->bgr(context->hand_roi);
        hd_obj.test(img, knn, 0.45);
        
        cv::addWeighted(img2, 0.7, hd_obj._ppr, 0.3, 0, img2);
        cv::cvtColor(img2, img2, CV_BGR2RGB);
        img2.copyTo(context->hand_ppr(context->hand_roi));
        
        // covert object mask to gray mask
        cv::cvtColor(hd_obj._ppr, context->obj_mask, CV_BGR2GRAY);
        context->obj_mask = context->obj_mask > 0;

        cv::Mat common_region;
        cv::Mat temp;
        cv::bitwise_not(context->obj_mask, temp);
        cv::bitwise_and(context->hand_mask, temp, context->hand_mask);
        */
        
        // calculate center of object
        float cx = 0.0f;
        float cy = 0.0f;
        float wz = 0.0;
        int hand_pix_cnt = 0;
        
        for(int i = context->hand_roi.y; i < context->hand_roi.y + context->hand_roi.height-1; i++)
        {
            for(int j = context->hand_roi.x; j < context->hand_roi.x + context->hand_roi.width-1; j++)
            {
                // probability
                cv::Vec3b p = hd_obj._ppr.at<cv::Vec3b>(i-context->hand_roi.y,j-context->hand_roi.x);
                if(p[1] > 0){
                    cy += i;
                    cx += j;
                    // note: flipped
                    wz += context->kinect->getDistanceAt(context->width - j, i);
                    hand_pix_cnt++;
                }
            }
        }
        if(hand_pix_cnt>20){
            cx = cx/hand_pix_cnt;
            cy = cy/hand_pix_cnt;
            wz = wz/hand_pix_cnt;
            
            context->center_of_obj[0] = cx;
            context->center_of_obj[1] = cy;
            context->center_of_obj[2] = wz;
            
            context->is_obj_detected = true;
        }else{
            context->is_obj_detected = false;
        }
        
        // calculate center of hand
        cx = 0.0f;
        cy = 0.0f;
        wz = 0.0;
        hand_pix_cnt = 0;
        
        for(int i = context->hand_roi.y; i < context->hand_roi.y + context->hand_roi.height-1; i++)
        {
            for(int j = context->hand_roi.x; j < context->hand_roi.x + context->hand_roi.width-1; j++)
            {
                // probability
                //cv::Vec3b p = hd_hand._ppr.at<cv::Vec3b>(i-context->hand_roi.y,j-context->hand_roi.x);
                unsigned char p = context->hand_mask.at<unsigned char>(i-context->hand_roi.y,j-context->hand_roi.x);
                
                if(p > 0){
                    cy += i;
                    cx += j;
                    // note: flipped
                    wz += context->kinect->getDistanceAt(context->width - j, i);
                    hand_pix_cnt++;
                }
            }
        }
        if(hand_pix_cnt>50){
            cx = cx/hand_pix_cnt;
            cy = cy/hand_pix_cnt;
            wz = wz/hand_pix_cnt;
            
            context->center_of_hand[0] = cx;
            context->center_of_hand[1] = cy;
            context->center_of_hand[2] = wz;
            
            context->is_hand_detected = true;
        }else{
            context->is_hand_detected = false;
        }
        
        
    }
}

void HandDetectorWrapper::start_saving(){
    save_cnt = 0;
}

/*
    save detector output for grasp classification training
 */
int HandDetectorWrapper::save(AppContext *context){
    
    const int max_int_frames = 2;
    static int int_frame_cnt = 0;
    
    // sampling every max_int_frames
    int_frame_cnt++;
    if(int_frame_cnt < max_int_frames){
        if(save_cnt == 0) return 1;
        else return save_cnt;
    }else{
        int_frame_cnt = 0;
    }
    
    if(save_cnt >= num_grasp_imgs){
        return 0;
    }
    
    char buf[128];
    sprintf(buf, "mask%08d.jpg", save_cnt);
    string mask_file = buf;
    mask_file = grasp_mask_prefix + mask_file;
    
    sprintf(buf, "%08d.jpg", save_cnt);
    string img_file = buf;
    img_file = grasp_img_prefix + img_file;
    
    sprintf(buf, "depth%08d.jpg", save_cnt);
    string depth_file = buf;
    depth_file = grasp_depth_prefix + depth_file;
    
    cv::Mat mat;
    context->bgr(context->hand_roi).copyTo(mat);
    cv::imwrite(img_file, mat);
    
    //cv::cvtColor(hd._ppr, mat, CV_BGR2GRAY);
    //mat = mat > 0;
    //cv::imwrite(mask_file, mat);
    cv::imwrite(mask_file, context->hand_mask);
    
    cv::imwrite(depth_file, context->depth(context->hand_roi));
    
    save_cnt++;
    
    return save_cnt;
    //return save_cnt > num_grasp_imgs;
}


 /*
void HandDetectorWrapper::update(AppContext *context){
    if(context->is_frame_new){
        cv::Mat img;
        cv::resize(context->bgr, img, cv::Size(), 0.5, 0.5);
        hd.test(img, knn);
        
        cv::addWeighted(img, 0.7, hd._ppr, 0.3, 0, context->hand_ppr);
        cv::resize(context->hand_ppr, context->hand_ppr, cv::Size(context->height, context->width), 0, 0, INTER_LINEAR);
        
        // calculate center of hand
        float cx = 0.0f;
        float cy = 0.0f;
        float wz = 0.0;
        int hand_pix_cnt = 0;
        for(int i = 0; i < hd._ppr.rows; i++)
        {
            for(int j = 0; j < hd._ppr.cols; j++)
            {
                // probability
                cv::Vec3b p = hd._ppr.at<cv::Vec3b>(i,j);
                if(p[0] > 0){
                    cy += i;
                    cx += j;
                    wz += context->kinect->getDistanceAt(2*j, 2*i);
                    hand_pix_cnt++;
                }
            }
        }
        if(hand_pix_cnt){
            cx = cx/hand_pix_cnt;
            cy = cy/hand_pix_cnt;
            wz = wz/hand_pix_cnt;
            
            // 2x scale
            cx *= 2;
            cy *= 2;
        
            context->center_of_hand[0] = cx;
            context->center_of_hand[1] = cy;
            context->center_of_hand[2] = wz;
        }
    }
}
*/
