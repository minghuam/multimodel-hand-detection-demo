#include "torso_com.h"
#include <climits>

void TorsoCom::setup(AppContext *context){
    
    float near_clip = context->kinect->getNearClipping();   // 255 for depth image
    float far_clip = context->kinect->getFarClipping();     // 0 for depth image
    
    float head_distance_range = 100;    // 10cm
    float ratio = (255 - 0)/(far_clip - near_clip);
    
    _head_depth_range = (int)(head_distance_range*ratio);
    
    _min_head_pix_cnt = 10;
    
}

void TorsoCom::update(AppContext *context){

    int left = context->body_roi.x;
    int right = context->body_roi.x + context->body_roi.width;
    int top = context->body_roi.y;
    int bottom = context->body_roi.y + context->body_roi.height;
    
    // get the closest point to camera in body roi
    int near = 0;
    for(int i=top; i<bottom; i++){
        for(int j=left; j<right; j++){
            int d = context->depth.at<unsigned char>(i,j);
            near = d > near ? d : near;
        }
    }
    
    // set up range
    int far = near - _head_depth_range;
    if(far < 0) far = 0;
    
    // calc center of mass
    int pix_cnt = 0;
    float cx = 0.0f;
    float cy = 0.0f;
    float wz = 0.0f;
    for(int i=top; i<bottom; i++){
        for(int j=left; j<right; j++){
            int d = context->depth.at<uchar>(i,j);
            if(d > far && d <= near){
                cx += j;
                cy += i;
                // note: flipped
                wz += context->kinect->getDistanceAt(context->width - j, i);
                pix_cnt++;
            }
        }
    }
    
    if(pix_cnt > _min_head_pix_cnt){
        cx = cx/pix_cnt;
        cy = cy/pix_cnt;
        wz = wz/pix_cnt;
        context->center_of_mass[0] = cx;
        context->center_of_mass[1] = cy;
        context->center_of_mass[2] = wz;
        
    }
}

/*

void TorsoCom::initialize(int no_of_bins)
{
    bins = no_of_bins;
}

void TorsoCom::threshold_depth()
{
    
    int bin_width=(256/bins);
    int index;
    for(int j=0; j<bins; j++)
    {
        depth_hist[j] = 0;
    }
    
    threshDepth = Mat::zeros(depth.rows,depth.cols,CV_8U);
    
    for(int i=0; i<depth.rows; i++)
    {
        for(int j=0; j<depth.cols; j++)
        {
            index =  (int)floor(depth.at<uchar>(i,j)/bin_width);
            depth_hist[index] = depth_hist[index] + 1;
        }
    }
    
    int max_depth_index = 10;
    int k=11;
    int flag = 0;
    while(k<bins-1)
    {
        //cout<<depth_hist[k]<<endl;
        if(depth_hist[k]>=depth_hist[max_depth_index])
        {
            max_depth_index = k;
            if(depth_hist[k+1]<depth_hist[k])
            {
                flag = 1;
                break;
            }
            
        }
        
        k=k+1;
    }
    

    nearThreshold = (max_depth_index)*bin_width;
    //if(max_depth_index==64)
    farThreshold = (max_depth_index + 1)*bin_width-1;
    //else
    //     farThreshold = (max_depth_index + 1)*4+4;
    //cout<<"****"<<nearThreshold<<" "<<farThreshold<<"***"<<endl;
    
    for(int i=0; i<depth.rows; i++)
    {
        for(int j=0; j<depth.cols; j++)
        {
            if(depth.at<uchar>(i,j)>nearThreshold && depth.at<uchar>(i,j)<farThreshold)
                threshDepth.at<uchar>(i,j)=255;
            else
                threshDepth.at<uchar>(i,j)=0;
        }
    }
    
    
}

void TorsoCom::computeCOM2(){
    
    int near = 255;
    int far = 200;
    
    int pix_cnt = 0;
    for(int i = 0; i < depth.rows; i++)
    {
        for(int j = 0; j < depth.cols; j++)
        {
            int d = depth.at<uchar>(i,j);
            if(d >= far && d <= near){
                    COM_x += j;
                    COM_y += i;
                    pix_cnt++;
            }
        }
    }
    
    if(pix_cnt){
        COM_x = COM_x/pix_cnt;
        COM_y = COM_y/pix_cnt;
        COM_z= (float)depth.at<unsigned char>((int)COM_x, (int)COM_y);
    }
}

void TorsoCom::computeCOM()
{
	COM_x = 0;
    COM_y = 0;
    COM_z = 0;
	int nop = 0;
	int eps = 0.001;
    
    for(int i = 0; i < depth.rows; i++)
	{
	    for(int j = 0; j < depth.cols; j++)
	    {
            if(threshDepth.at<uchar>(i,j) ==255)
            {
                COM_x = COM_x + j;
                COM_y = COM_y + i;
                COM_z = COM_z + depth.at<uchar>(i,j);
                nop = nop+1;
            }
        }
    }
    
    
    COM_x = COM_x/nop+eps;
    COM_y = COM_y/nop+eps;
	COM_z = COM_z/nop +eps;
    
    //cout << nop << "," << COM_x << "," << COM_y << "," << COM_z << endl;
    
}
*/
