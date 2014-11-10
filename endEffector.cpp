//
//  endEffector.cpp
//  QoLT
//
//  Created by Akanksha Saran on 10/21/14.
//
//

#include "endEffector.h"

void endEffector::initialize(const char* xml_1, const char* xml_2)
{
   
    //string filename = "medium_wrap.xml";
    //_classifier1.load( filename.c_str());
   //string filename2 = "precision_disc.xml";
    //_classifier2.load( filename2.c_str());

    cout << xml_1 << endl;
    cout << xml_2 << endl;
    
    _classifier1.load(xml_1);
    _classifier2.load(xml_2);
    
    label_distr[0] = 0.0f;
    label_distr[1] = 0.0f;
    
}

void endEffector::getPatch(Mat &bin)
{
    // commented by minghuam, bin is already grayscale
    //cvtColor( bin, _bin, COLOR_BGR2GRAY );  //bin is colored img
    //_bin = _bin>70;                         //_bin is grayscale
    
    _bin = bin > 70;
    
    Mat bin4contour;
    bin4contour = _bin.clone();                 //bin4contour is grayscale
    Mat Blur_bin = Mat::zeros(100,100,CV_8U);    //Blur_bin is grayscale
    vector<Vec4i> hierarchy;
    vector<vector<Point> > hand_contours;
    findContours(bin4contour, hand_contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    double max_area1 = 0; int index1=-1;
    
    for(int i=0;i<hand_contours.size();i++)
    {
        double area = contourArea(hand_contours[i]);
        if (area>max_area1)
        {
            max_area1 = area;
            index1 = i;
        }
    }
    
    
    double max_area2 = 0; int index2=-1;
    for(int i=0;i<hand_contours.size();i++)
    {
        if(i==index1)
            continue;
        double area = contourArea(hand_contours[i]);
        if (area>max_area2)
        {
            max_area2 = area;
            index2 = i;
        }
    }
    
    
    vector<int> x;
    vector<int> y;
    if (index1!=-1 && max_area1>0)
    {
        for(int j=0;j<hand_contours[index1].size();j++)
        {
            x.push_back(hand_contours[index1][j].x);
            y.push_back(hand_contours[index1][j].y);
        }
    }

    int x_sum=0;
    int y_sum=0;
    for(int j=0; j<x.size(); j++)
    {
        x_sum += x[j];
        y_sum += y[j];
    }
    
    int x_avg=0;
    int y_avg=0;
    int w_neg = 0;
    int w_pos = 0;
    int h_neg = 0;
    int h_pos = 0;
    int w = 0;
    int h = 0;
    
    if(x.size()!=0 && y.size()!=0)
    {       
        x_avg = (int) (x_sum/x.size());
        y_avg = (int) (y_sum/y.size());
        
        vector<int> x_diff;
        vector<int> y_diff;
        for(int j=0; j<x.size(); j++)
        {
            x_diff.push_back(x[j]);
            y_diff.push_back(y[j]);
        }
        
        sort(x_diff.begin(), x_diff.end());
        sort(y_diff.begin(), y_diff.end());
        
        w_neg = x_diff[0];
        w_pos = x_diff[x.size()-1];
        h_neg = y_diff[0];
        h_pos = y_diff[y.size()-1];
        w = w_pos - w_neg + 1;
        h = h_pos - h_neg + 1;

	if(w_neg-2>0) {w_neg=w_neg-2; w=w+2;}
	if(w_pos+2<_bin.cols-1) {w_pos=w_pos+2; w=w+2;}
	if(h_neg-2>0) {h_neg=h_neg-2; h=h+2;}
	if(h_pos+2<_bin.rows-1) {h_pos=h_pos+2; h=h+2;}

	Rect boundRect;
        boundRect = Rect(w_neg, h_neg, w, h);
        
        int tl_x=w_neg;
        int tl_y=h_neg;
        int size_x = w;
        int size_y = h;
        
        Rect roi_patch = Rect(tl_x, tl_y, size_x, size_y);
        Mat patch = _bin(roi_patch).clone();                                     //patch is grayscale
        
        Size c;
        c.height =100;
        c.width = 100;
        resize(patch, _patch, c, 0,0, INTER_LINEAR);  
        

    } 
    else
    {
        _patch = Mat::zeros(100,100,CV_8U); 
   
    }

    //cout<<"no of pixels: "<<count_whitePixels(_patch)<<endl;
    //imwrite("patch.jpg",_patch);
}

    


double endEffector::count_whitePixels(cv::Mat &patch)
{
    double c=0;
    for(int i=0; i<patch.rows; i++)
    {
        for(int j=0; j<patch.cols; j++)
        {
            uchar intensity = patch.at<uchar>(i,j);
            if(intensity==255)
                c++;
        }
    }
    return c;
}

vector<double> endEffector::calc_moments(Mat &patch)
{
    Mat gray_patch;
    cvtColor( patch, gray_patch, COLOR_BGR2GRAY );
    double M[4][4];
    double sum = 0;
    for(int i=0; i<patch.rows; i++)
    {
        for(int j=0; j<patch.cols; j++)
        {
            sum += gray_patch.at<uchar>(i,j);   //verify power order
        }
    }
    
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
            M[m][n] = 0;
            for(int i=0;i<patch.rows; i++)
            {
                for(int j=0; j<patch.cols; j++)
                {
                    M[m][n] += std::pow((double)j,(double)m)*std::pow((double)i,(double)n)*gray_patch.at<uchar>(i,j);   //verify power order
                }
            }
            M[m][n] = M[m][n]/sum;
        }
    }
    
    double x_avg = M[1][0]/M[0][0];
    double y_avg = M[0][1]/M[0][0];
    double nu[4][4]= {0};
    nu[0][0] = M[0][0];
    nu[0][1] = 0;
    nu[1][0] = 0;
    nu[1][1] = M[1][1] - x_avg*M[0][1];
    nu[2][0] = M[2][0] - x_avg*M[1][0];
    nu[0][2] = M[0][2] - y_avg*M[0][1];
    nu[2][1] = M[2][1] - 2*x_avg*M[1][1] - y_avg*M[2][0] + 2*x_avg*x_avg*M[0][1];
    nu[1][2] = M[1][2] - 2*y_avg*M[1][1] - x_avg*M[0][2] + 2*y_avg*y_avg*M[1][0];
    nu[3][0] = M[3][0] - 3*x_avg*M[2][0] + 2*x_avg*x_avg*M[1][0];
    nu[0][3] = M[0][3] - 3*y_avg*M[0][2] + 2*y_avg*y_avg*M[0][1];
    
    
    double eta[4][4]={0};
    eta[2][0] = nu[2][0]/pow(nu[0][0],2);
    eta[0][2] = nu[0][2]/pow(nu[0][0],2);
    eta[1][2] = nu[1][2]/pow(nu[0][0],2);
    eta[2][1] = nu[2][1]/pow(nu[0][0],2);
    eta[3][0] = nu[3][0]/pow(nu[0][0],2);
    eta[0][3] = nu[0][3]/pow(nu[0][0],2);
    eta[1][1] = nu[1][1]/pow(nu[0][0],2);
    
    
    
    vector<double> I;
    I.push_back(eta[2][0]+eta[0][2]);
    I.push_back(pow(eta[2][0]-eta[0][2],2)+4*pow(eta[1][1],2));
    double x = eta[1][1]*(pow(eta[3][0]+eta[1][2],2)-pow(eta[0][3]+eta[2][1],2)) - (eta[2][0]-eta[0][2])*(eta[3][0]+eta[1][2])*(eta[0][3]+eta[2][1]);
    I.push_back(x);
    I.push_back(pow(eta[3][0]+eta[1][2],2)+pow(eta[2][1]+eta[0][3],2));
    x = (eta[3][0]-3*eta[1][2])*(eta[3][0]+eta[1][2])*(pow(eta[3][0]+eta[1][2],2)-3*pow(eta[2][1]+eta[0][3],2)) + (3*eta[2][1]-eta[0][3])*(eta[2][1]+eta[0][3])*(3*pow(eta[3][0]+eta[1][2],2)-pow(eta[2][1]+eta[0][3],2));
    I.push_back(x);
    x = (eta[2][0]-eta[0][2])*(pow(eta[3][0]+eta[1][2],2)-pow(eta[2][1]+eta[0][3],2)) + 4*eta[1][1]*(eta[3][0]+eta[1][2])*(eta[2][1]+eta[0][3]);
    I.push_back(x);
    x = (3*eta[2][1]-eta[0][3])*(eta[3][0]+eta[1][2])*(pow(eta[3][0]+eta[1][2],2)-3*pow(eta[2][1]+eta[0][3],2)) - (eta[3][0]-3*eta[1][2])*(eta[2][1]+eta[0][3])*(3*pow(eta[3][0]+eta[1][2],2) - pow(eta[2][1]+eta[0][3],2));
    I.push_back(x);
    
    return I;
}


vector<double> endEffector::calc_haar(Mat &patchGray)
{
    
    Size b;
    b.height = 25;
    b.width = 25;
    Mat small_box=Mat::zeros(25,25,CV_8U);
    resize(patchGray, small_box, b, 0,0, INTER_LINEAR);
    
    int Rect_x[5][5];
    int Rect_y[5][5];
    double total_white = count_whitePixels(small_box);
    int r_size = 5;
    vector<double> rect_prop;
    
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<5; j++)
        {
            Rect_x[i][j] = j*5;
            Rect_y[i][j] = i*5;
        }
    }
     
    int x, y;
    double val;
    double sum = 0;
    int rect_count = 0;
    
    for(int i=1; i<6; i++)
    {
        for(int j=1; j<6; j++)
        {
            for(int m=0; m<5; m++)
            {
                for(int n=0; n<5; n++)
                {
                    x = Rect_x[m][n];
                    y = Rect_y[m][n];
                    if((x+i*r_size-1 < small_box.cols) && (y+j*r_size-1<small_box.rows))
                    {
                        rect_count++;
                        Rect roi = Rect( x, y, r_size*i, r_size*j );
                        Mat rect = small_box(roi).clone();
                        val = count_whitePixels(rect)/total_white;
                        rect_prop.push_back(val);
                    }
                }
            }	
        }
    }
    
    return rect_prop;
}


void endEffector::computeFeatures()
{
    
    Features = Mat::zeros(1,232,CV_32F);
    patchGray =_patch>70;
    if(count_whitePixels(patchGray)>0)
    {
        
        Moments m = moments(patchGray, true);
        double hu[7];
        HuMoments(m, hu);
        vector<double> moment_feat;
        
        for(int i=0; i<7; i++){
            moment_feat.push_back(hu[i]);
        }
        
        vector<double> rect_feat = calc_haar(patchGray);   
        
        vector<double> feat;
        //combining different feature vectors
        feat.reserve( moment_feat.size() ); // preallocate memory
        feat.insert( feat.end(), moment_feat.begin(), moment_feat.end() );
        feat.insert( feat.end(), rect_feat.begin(), rect_feat.end() );        
        
        for(int i=0; i<feat.size(); i++)
        {
            Features.at<float>(0,i) = feat[i];
            
            //cout << Features.at<float>(0,i) << "," << feat[i]  << "," << feat.size() << endl;
        }
        
    }
  
}

int endEffector::test()
{
    
    static vector<float> track_temp1;
    static vector<float> track1;
    static vector<float> track_temp2;
    static vector<float> track2;
    int N ;
    int f = 10;
    
    static int label_old = 0;
    //static float label_distr_old[2] = {0};
    
    //float label_distr[2] = {0};
    
    Mat prob;
    prob = Mat::zeros(1, 1, 5);
    
    prob.at<float>(0,0) =  _classifier1.predict_prob( Features );
    //label_distr.push_back(prob.at<float>(0,0));
    label_distr[0] = prob.at<float>(0,0);
    
    prob.at<float>(0,0) =  _classifier2.predict_prob( Features );
    //label_distr.push_back(prob.at<float>(0,0));
    label_distr[1] = prob.at<float>(0,0);
    
    track1.push_back(label_distr[0]);
    track_temp1.push_back(label_distr[0]);
    track2.push_back(label_distr[1]);
    track_temp2.push_back(label_distr[1]);
    
    if(track1.size()>f)
    {
        track1.erase (track1.begin());
        track2.erase (track2.begin());
        track_temp1.erase (track_temp1.begin());
        track_temp2.erase (track_temp2.begin());
    
        N = track1.size();
        
        for(int j=N-f-1; j<N-1; j++)
        {
                track_temp1[N-1] += track1[j];
                track_temp2[N-1] += track2[j];
        }
        track_temp1[N-1] = track_temp1[N-1]/(f+1);
        track_temp2[N-1] = track_temp2[N-1]/(f+1);
       
        
        label_distr[0] = track_temp1[N-1];
        label_distr[1] = track_temp2[N-1];
        
    }
    
    if(count_whitePixels(_patch)==0){
        track1.erase (track1.begin(),track1.end());
        track2.erase (track2.begin(),track2.end());
        track_temp1.erase (track_temp1.begin(),track_temp1.end());
        track_temp2.erase (track_temp2.begin(),track_temp2.end());
    }
    
    
    
    //label_distr[0] = label_distr[0] * 0.7f + label_distr_old[0] * (0.3f);
    //label_distr[1] = label_distr[1] * 0.7f + label_distr_old[1] * (0.3f);
    
    //label_distr_old[0] = label_distr[0];
    //label_distr_old[1] = label_distr[1];
    
    //cout<<"prob distr "<< label_distr[0] <<" "<<label_distr[1]<<endl;
    
    float threshold = 0.5f;
    if(label_distr[0] > threshold){
        if(label_distr[1] <= threshold){
            grasp_label= 1;
        }else{
            if(label_distr[0] == label_distr[1]){
                grasp_label = label_old;
            }else{
                grasp_label = label_distr[0] > label_distr[1] ? 1 : 2;
            }
        }
    }else{
        grasp_label = label_distr[1] > threshold ? 2 : 0;
    }
    
    /*
    grasp_label = label_distr[0]>label_distr[1] ? 1:2;
    if(label_distr[0]<0.5 && label_distr[1]<0.5)
        grasp_label = 0;
    */
     
    //cout<<"grasp label: "<<grasp_label<<endl;
    
    label_old = grasp_label;
    
    return grasp_label;
    
}

/*
int endEffector::test()
{
    vector<float>  		label_distr;
    
    Mat prob;
    prob = Mat::zeros(1, 1, 5);

    prob.at<float>(0,0) =  _classifier1.predict_prob( Features );
    label_distr.push_back(prob.at<float>(0,0));
    prob.at<float>(0,0) =  _classifier2.predict_prob( Features );
    label_distr.push_back(prob.at<float>(0,0));
    
    cout<<"prob distr "<< label_distr[0]<<" "<<label_distr[1]<<endl;
    
    grasp_label = label_distr[0]>label_distr[1] ? 1:2;
    if(label_distr[0]<0.5 && label_distr[1]<0.5)
     grasp_label = 0;

    //cout<<"grasp label: "<<grasp_label<<endl;
    
    return grasp_label;

}
 */

