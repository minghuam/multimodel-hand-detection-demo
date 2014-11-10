//
//  demo_ui.cpp
//  HandDemo
//
//  Created by Akanksha Saran on 10/24/14.
//
//

#include "demo_ui.h"

void DemoUI::setup(AppContext *context){

    
    // of image allocation
    _hand_of.allocate(context->width, context->height, OF_IMAGE_COLOR);

    _hand_of.clear();
    
    // set up viewports
    int margin = 10;
    const int rows = 1;
    const int cols = 2;
    int view_width = (ofGetWidth() - margin*cols)/cols;
    float img_ratio = (float)context->height/context->width;
    int view_height = view_width*img_ratio;
    
    int title_height = 100;
    int x = margin;
    int y = title_height + margin;
    ofRectangle rects[rows*cols];
    for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            rects[row*cols + col] = ofRectangle(x,y,view_width, view_height);
            x += (view_width + margin);
        }
        x = margin;
        y += (view_height + margin);
    }
    
    _title_viewport = ofRectangle(0,0,ofGetWidth(), title_height);
    _hand_viewport = rects[0];
    _centers_viewport = rects[1];

    // 3d mesh for point cloud
    _3d_mesh.setMode(OF_PRIMITIVE_POINTS);
    
    _z_offset = -800;
}

void DemoUI::update(AppContext *context){
    if(context->is_frame_new){
        // covert to ofImage for UI
        if(context->state != APP_STATE_RUNNING){
            ofxCv::toOf(context->rgb, _hand_of);
            _hand_of.update();
        }else{
            ofxCv::toOf(context->hand_ppr, _hand_of);
            _hand_of.update();
        }
        
        _3d_mesh.clear();
        int step = 2;
        for(int row = 0; row < context->height; row += step){
            for(int col = 0; col < context->width; col += step){
                ofVec3f pos = context->kinect->getWorldCoordinateAt(col,row);
                pos = ofVec3f(pos[0], pos[1], pos[2] + _z_offset);
                _3d_mesh.addVertex(pos);
                _3d_mesh.addColor(context->kinect->getColorAt(col, row));
            }
        }
    }
}

void DemoUI::draw(AppContext *context){
    
    static vector<float> track_x;
    static vector<float> track_y;
    static vector<float> track_temp_x;
    static vector<float> track_temp_y;
    
    ofBackground(0);
    
    ofSetColor(255,255,255);
    _hand_of.draw(_hand_viewport);
    ofSetColor(0,255,0,60);

    context->artk->draw(_hand_viewport.x, _hand_viewport.y, _hand_viewport.width, _hand_viewport.height);
    
    // draw head and hand centers on hand image
    ofFill();
    float scale = _hand_viewport.width/context->width;
    if(context->is_hand_detected){
        ofSetColor(255, 0, 0);
        ofCircle(ofPoint(_hand_viewport.x + context->center_of_hand[0]*scale, _hand_viewport.y + context->center_of_hand[1]*scale), 5);
    }
    if(context->is_obj_detected){
        ofSetColor(0, 0, 255);
        ofCircle(ofPoint(_hand_viewport.x + context->center_of_obj[0]*scale, _hand_viewport.y + context->center_of_obj[1]*scale), 5);
    }
    
    ofSetLineWidth(3);
    track_x.push_back(_hand_viewport.x + context->center_of_hand[0]*scale);
    track_y.push_back(_hand_viewport.y + context->center_of_hand[1]*scale);
    track_temp_x.push_back(_hand_viewport.x + context->center_of_hand[0]*scale);
    track_temp_y.push_back(_hand_viewport.y + context->center_of_hand[1]*scale);
    
    
    if(track_x.size()>=5)
    { for(int i=4; i<track_x.size(); i++)
	   {
           for(int j=i-4; j<i; j++)
           {
               track_temp_y[i] += track_y[j];
               track_temp_x[i] += track_x[j];
           }
           track_temp_y[i] = track_temp_y[i]/5;
           track_temp_x[i] = track_temp_x[i]/5;
       }
        
    }
    
    ofEnableAlphaBlending();
    
    if(track_x.size()>0)
    {
        if(track_x.size()<=10)
        {
            for(int i=2; i<track_x.size(); i++)
            {
                ofSetColor(10-i-1,255,255,i*25);
                ofLine(track_temp_x[i-1],track_temp_y[i-1], track_temp_x[i],track_temp_y[i]);
                
            }
        }
        else
        {
            for(int i=track_x.size()-10; i<track_x.size(); i++)
            {
                ofSetColor(track_x.size()-i-1,255,255,i*25);
                ofLine(track_temp_x[i-1],track_temp_y[i-1], track_temp_x[i],track_temp_y[i]);
                
            }
        }
    }
    
    
    if(track_x.size()>11)
    {
        track_x.erase (track_x.begin());
        track_y.erase (track_y.begin());
        track_temp_x.erase (track_temp_x.begin());
        track_temp_y.erase (track_temp_y.begin());
    }
    
    // draw head and hand centers in 3D
    ofVec3f pos1 = context->kinect->getWorldCoordinateAt(context->width - context->center_of_hand[0], context->center_of_hand[1], context->center_of_hand[2]);
    ofVec3f pos2 = context->kinect->getWorldCoordinateAt(context->width - context->center_of_mass[0],context->center_of_mass[1],context->center_of_mass[2]);
    ofVec3f pos3 = context->kinect->getWorldCoordinateAt(context->width - context->center_of_obj[0],context->center_of_obj[1],context->center_of_obj[2]);
    pos1 = ofVec3f(pos1[0], pos1[1], pos1[2] + _z_offset);
    pos2 = ofVec3f(pos2[0], pos2[1], pos2[2] + _z_offset);
    pos3 = ofVec3f(pos3[0], pos3[1], pos3[2] + _z_offset);
    draw_ptcloud_centers(context, _centers_viewport, pos1, pos2, pos3);
}

void DemoUI::draw_ptcloud_centers(AppContext *context, ofRectangle &viewport, const ofVec3f &pt1, const ofVec3f &pt2, const ofVec3f &pt3){
    
    ofSetColor(255, 255, 255);
    ofNoFill();
    ofRect(viewport);
    
    ofEnableDepthTest();
    
    _ptcloud_camera.setDistance(1000);
    
    // draw point cloud
    ofEnableDepthTest();
    glEnable(GL_POINT_SMOOTH);
    _ptcloud_camera.begin(viewport);
    ofDrawAxis(100);
    ofScale(-1,-1,-1);
    glPointSize(1);
    _3d_mesh.draw();
    
    ofDisableDepthTest();
    
    // draw centers
    ofFill();
    if(context->is_hand_detected){
        ofSetColor(255,0,0);
        ofDrawSphere(pt1[0], pt1[1], pt1[2], 10);
        ofSetColor(255,255,0);
        ofLine(pt1[0], pt1[1], pt1[2], pt2[0], pt2[1], pt2[2]);
        
    }
    ofSetColor(0,255,0);
    ofDrawSphere(pt2[0], pt2[1], pt2[2], 10);
    if(context->is_obj_detected){
        ofSetColor(0,0,255);
        ofDrawSphere(pt3[0], pt3[1], pt3[2], 10);
        ofSetColor(255, 255, 0);
        if(context->is_hand_detected){
            ofLine(pt1[0], pt1[1], pt1[2], pt3[0], pt3[1], pt3[2]);}
    }
    
    _ptcloud_camera.end();
    
}