//
//  ui.cpp
//  HandDemo
//
//  Created by Minghuang Ma on 10/22/14.
//
//

#include "ui.h"


void UI::setup(AppContext *context){
    
    // of image allocation
    _rgb_of.allocate(context->width, context->height, OF_IMAGE_COLOR);
    _depth_of.allocate(context->width, context->height, OF_IMAGE_GRAYSCALE);
    _mask_of.allocate(context->width, context->height, OF_IMAGE_GRAYSCALE);
    _hand_of.allocate(context->width, context->height, OF_IMAGE_COLOR);
    _raw_mask_of.allocate(context->width, context->height, OF_IMAGE_GRAYSCALE);
    
    // 4x2 tiles
    _train_tile_rows = 4;
    _train_tile_cols = 1;
    for(int i=0;i<_train_tile_rows*_train_tile_cols; i++){
        ofImage img;
        img.allocate(context->hand_roi.width, context->hand_roi.height, OF_IMAGE_COLOR);
        img.clear();
        _train_imgs_of.push_back(img);
        ofImage msk;
        msk.allocate(context->hand_roi.width, context->hand_roi.height, OF_IMAGE_GRAYSCALE);
        msk.clear();
        _train_msks_of.push_back(msk);
    }
    
    _rgb_of.clear();
    _depth_of.clear();
    _mask_of.clear();
    _hand_of.clear();
    
    // set up viewports
    int margin = 10;
    const int rows = 2;
    const int cols = 3;
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
    _rgb_viewport = rects[0];
    _depth_viewport = rects[1];
    _mask_viewport = rects[2];
    _hand_viewport = rects[3];
    _centers_viewport = rects[4];
    _grasp_viewport = rects[5];
    
    // phase 1: select roi for head
    // phase 2: select roi for hand
    // pahse 3: select roi for object
    _calibrate_phase = 3;
    
    // 3d mesh for point cloud
    _3d_mesh.setMode(OF_PRIMITIVE_POINTS);
    
    z_offset = -800;
}

void UI::update(AppContext *context){

    if(context->is_frame_new){
        
        // covert to ofImage for UI
        ofxCv::toOf(context->rgb, _rgb_of);
        ofxCv::toOf(context->depth, _depth_of);
        ofxCv::toOf(context->mask, _mask_of);
        ofxCv::toOf(context->hand_ppr, _hand_of);
        ofxCv::toOf(context->raw_mask, _raw_mask_of);
        
        if(context->state == APP_STATE_TRAIN_HANDMODEL){
            cv::Mat temp;
            
            cv::cvtColor(context->current_training_img, temp, CV_BGR2RGB);
            ofxCv::toOf(temp, _train_imgs_of[_tile_start_index]);
            _train_imgs_of[_tile_start_index].update();
            
            context->current_training_msk.copyTo(temp);
            ofxCv::toOf(temp, _train_msks_of[_tile_start_index]);
            _train_msks_of[_tile_start_index].update();
            
            _tile_start_index++;
            if(_tile_start_index > _train_tile_cols*_train_tile_rows-1){
                _tile_start_index = 0;
            }
        }
        
        _rgb_of.update();
        _depth_of.update();
        _mask_of.update();
        _hand_of.update();
        _raw_mask_of.update();

        _3d_mesh.clear();
        int step = 2;
        for(int row = 0; row < context->height; row += step){
            for(int col = 0; col < context->width; col += step){
                ofVec3f pos = context->kinect->getWorldCoordinateAt(col,row);
                pos = ofVec3f(pos[0], pos[1], pos[2] + z_offset);
                _3d_mesh.addVertex(pos);
                _3d_mesh.addColor(context->kinect->getColorAt(col, row));
            }
        }
    }
    
}

void UI::draw(AppContext *context){
    
    static vector<float> track_x;
    static vector<float> track_y;
    static vector<float> track_temp_x;
    static vector<float> track_temp_y;
    
    ofBackground(0);
    
    ofPath ring;
    
    ofSetColor(255,255,255);
    _rgb_of.draw(_rgb_viewport);
    _depth_of.draw(_depth_viewport);
    _hand_of.draw(_hand_viewport);
    
    //_raw_mask_of.draw(_grasp_viewport);
    
    // draw mask over rgb
    _rgb_of.draw(_mask_viewport);
    ofSetColor(0, 255, 0,60);
    _mask_of.draw(_mask_viewport);
    ofSetColor(255, 255, 255);
    ofRectangle roi_viewport(0,0,context->width, context->height);
    ofRectangle hand_rect(context->hand_roi.x, context->hand_roi.y, context->hand_roi.width, context->hand_roi.height);
    ofRectangle rect;
    transform_viewport(roi_viewport, _mask_viewport, hand_rect, rect);
    ofRect(_mask_viewport.x + rect.x, _mask_viewport.y + rect.y, rect.width, rect.height);
    
    // body
    ofSetColor(255, 255, 0);
    ofNoFill();
    ofRect(_rgb_viewport.x + rect.x, _rgb_viewport.y + rect.y, rect.width, rect.height);
    

    // draw head, hand and object centers on hand image
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
    
    // draw head, hand and object centers in 3D, flipped
    ofVec3f pos1 = context->kinect->getWorldCoordinateAt(context->width - context->center_of_hand[0], context->center_of_hand[1], context->center_of_hand[2]);
    ofVec3f pos2 = context->kinect->getWorldCoordinateAt(context->width - context->center_of_mass[0],context->center_of_mass[1],context->center_of_mass[2]);
    ofVec3f pos3 = context->kinect->getWorldCoordinateAt(context->width - context->center_of_obj[0],context->center_of_obj[1],context->center_of_obj[2]);
    pos1 = ofVec3f(pos1[0], pos1[1], pos1[2] + z_offset);
    pos2 = ofVec3f(pos2[0], pos2[1], pos2[2] + z_offset);
    pos3 = ofVec3f(pos3[0], pos3[1], pos3[2] + z_offset);
    draw_ptcloud_centers(context, _centers_viewport, pos1, pos2, pos3);
    
    draw_roi(context);
 
    // draw object and grasp
    draw_object_grasp(context, _hand_viewport);

    draw_training_tiles(context);
    
    
    track_x.push_back(_hand_viewport.x + context->center_of_hand[0]*scale);
    track_y.push_back(_hand_viewport.y + context->center_of_hand[1]*scale);
    track_temp_x.push_back(_hand_viewport.x + context->center_of_hand[0]*scale);
    track_temp_y.push_back(_hand_viewport.y + context->center_of_hand[1]*scale);
    
    ofSetLineWidth(3);
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
                ofSetColor(10-i-1,255,255,25*i);
                ofLine(track_temp_x[i-1],track_temp_y[i-1], track_temp_x[i],track_temp_y[i]);
                
            }
        }
        else
        {
            for(int i=track_x.size()-10; i<track_x.size(); i++)
            {
                ofSetColor(track_x.size()-i-1,255,255,25*i);
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
}

void UI::draw_ptcloud_centers(AppContext *context, ofRectangle &viewport, const ofVec3f &pt1, const ofVec3f &pt2, const ofVec3f &pt3){

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
        ofLine(pt1[0], pt1[1], pt1[2], pt3[0], pt3[1], pt3[2]);
    }
    
    _ptcloud_camera.end();
    
}

void UI::draw_object_grasp(AppContext *context, ofRectangle &viewport){
    context->artk->draw(viewport.x, viewport.y, viewport.width, viewport.height);
}

void UI::draw_training_tiles(AppContext *context){
    int w = context->hand_roi.width;
    int h = context->hand_roi.height;
    
    // hand imgs
    int i = _tile_start_index - 1;
    if(i < 0) i = _train_imgs_of.size() - 1;
    for(int row=0; row<_train_tile_rows; row++){
        for(int col=0; col<_train_tile_cols; col++){
            int x = _grasp_viewport.x + w*col;
            int y =_grasp_viewport.y + h*row;
            ofRect(x,y,w,h);
            _train_imgs_of[i].draw(x, y);
            i--;
            if(i < 0) i = _train_imgs_of.size() - 1;
        }
    }
    
    // hand masks
    i = _tile_start_index - 1;
    if(i < 0) i = _train_imgs_of.size() - 1;
    for(int row=0; row<_train_tile_rows; row++){
        for(int col=0; col<_train_tile_cols; col++){
            int x = _grasp_viewport.x + w*(col + _train_tile_cols);
            int y =_grasp_viewport.y + h*row;
            ofRect(x,y,w,h);
            _train_msks_of[i].draw(x, y);
            i--;
            if(i < 0) i = _train_imgs_of.size() - 1;
        }
    }
}


void UI::draw_roi(AppContext *context){
    
    ofRectangle roi_viewport(0,0,context->width, context->height);
    
    ofRectangle body_rect(context->body_roi.x, context->body_roi.y, context->body_roi.width, context->body_roi.height);
    ofRectangle hand_rect(context->hand_roi.x, context->hand_roi.y, context->hand_roi.width, context->hand_roi.height);
    ofRectangle obj_rect(context->obj_roi.x, context->obj_roi.y, context->obj_roi.width, context->obj_roi.height);
    
    ofRectangle rect;
    
    // body
    ofSetColor(255, 255, 0);
    ofNoFill();
    transform_viewport(roi_viewport, _rgb_viewport, body_rect, rect);
    ofRect(_rgb_viewport.x + rect.x, _rgb_viewport.y + rect.y, rect.width, rect.height);
    
    // hand
    ofSetColor(0, 255, 0);
    ofNoFill();
    transform_viewport(roi_viewport, _rgb_viewport, hand_rect, rect);
    ofRect(_rgb_viewport.x + rect.x, _rgb_viewport.y + rect.y, rect.width, rect.height);
    transform_viewport(roi_viewport, _mask_viewport, hand_rect, rect);
    ofRect(_mask_viewport.x + rect.x, _mask_viewport.y + rect.y, rect.width, rect.height);
    
    
    // object
    ofSetColor(0, 0, 255);
    ofNoFill();
    transform_viewport(roi_viewport, _rgb_viewport, obj_rect, rect);
    ofRect(_rgb_viewport.x + rect.x, _rgb_viewport.y + rect.y, rect.width, rect.height);
    transform_viewport(roi_viewport, _mask_viewport, obj_rect, rect);
    ofRect(_mask_viewport.x + rect.x, _mask_viewport.y + rect.y, rect.width, rect.height);
    
    ofSetColor(255, 255, 255);
}

bool UI::calibrate(AppContext *context){
    return (_calibrate_phase == 3);
}

void UI::start_calibrate(){
    _calibrate_phase = 0;
}

void UI::transform_viewport(ofRectangle &viewport1, ofRectangle &viewport2, ofRectangle rect1, ofRectangle &rect2){
    float scale_x = viewport2.width/viewport1.width;
    float scale_y = viewport2.height/viewport1.height;
    
    float x = viewport2.width*(rect1.x - 0)/viewport1.width;
    float y = viewport2.height*(rect1.y - 0)/viewport1.height;
    float w = viewport2.width*rect1.width/viewport1.width;
    float h = viewport2.height*rect1.height/viewport1.height;
    
    rect2 = ofRectangle(x,y,w,h);
}

//--------------------------------------------------------------
void UI::keyPressed(int key){
}

//--------------------------------------------------------------
void UI::keyReleased(int key){
}

//--------------------------------------------------------------
void UI::mouseMoved(AppContext *context, int x, int y ){
}

//--------------------------------------------------------------
void UI::mouseDragged(AppContext *context, int x, int y, int button){
    // traslate to viewport
    x = x - _rgb_viewport.x;
    y = y - _rgb_viewport.y;
    
    // clamp
    if(x < 0) x = 0;
    if(x > _rgb_viewport.width - 1) x = _rgb_viewport.width - 1;
    if(y < 0) y = 0;
    if(y > _rgb_viewport.height - 1) y = _rgb_viewport.height - 1;
    
    int left = min(x, _last_mouse_x);
    int top = min(y, _last_mouse_y);
    int w = abs(x - _last_mouse_x);
    int h = abs(y - _last_mouse_y);
    
    // map to roi
    ofRectangle rect1(left, top, w, h);
    ofRectangle rect2;
    ofRectangle roi_viewport(0,0,context->width, context->height);

    transform_viewport(_rgb_viewport, roi_viewport, rect1, rect2);
    
    if(_calibrate_phase == 0){
        context->body_roi = cv::Rect(rect2.x, rect2.y, rect2.width, rect2.height);
    }else if(_calibrate_phase == 1){
        context->hand_roi = cv::Rect(rect2.x, rect2.y, rect2.width, rect2.height);
    }else if(_calibrate_phase == 2){
        context->obj_roi = cv::Rect(rect2.x, rect2.y, rect2.width, rect2.height);
    }
}

//--------------------------------------------------------------
void UI::mousePressed(AppContext *context, int x, int y, int button){
    
    // traslate to viewport
    x = x - _rgb_viewport.x;
    y = y - _rgb_viewport.y;
    
    // clamp
    if(x < 0) x = 0;
    if(x > _rgb_viewport.width - 1) x = _rgb_viewport.width - 1;
    if(y < 0) y = 0;
    if(y > _rgb_viewport.height - 1) y = _rgb_viewport.height - 1;
    
    _last_mouse_x = x;
    _last_mouse_y = y;
}

//--------------------------------------------------------------
void UI::mouseReleased(AppContext *context, int x, int y, int button){
    
    if(_calibrate_phase == 0){
        _calibrate_phase = 1;
    }else if(_calibrate_phase == 1){
        _calibrate_phase = 2;
    }else if(_calibrate_phase == 2){
        _calibrate_phase = 3;
    }
    
    if(context->body_roi.width * context->body_roi.height == 0){
        context->body_roi = cv::Rect(0,0,context->width, context->height);
    }
    if(context->hand_roi.width * context->hand_roi.height == 0){
        context->hand_roi = cv::Rect(0,0,context->width, context->height);
    }
    if(context->obj_roi.width * context->obj_roi.height == 0){
        context->obj_roi = cv::Rect(0,0,context->width, context->height);
    }
}









