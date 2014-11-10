//
//  ui.h
//  HandDemo
//
//  Created by Minghuang Ma on 10/22/14.
//
//

#ifndef HandDemo_ui_h
#define HandDemo_ui_h

#include "ofMain.h"
#include "app_context.h"
#include "ofxCv.h"

class UI{
    
private:
    
    ofImage _rgb_of;
    ofImage _depth_of;
    ofImage _hand_of;
    
    ofImage _mask_of;
    ofImage _raw_mask_of;
    
    ofRectangle _title_viewport;
    ofRectangle _rgb_viewport;
    ofRectangle _depth_viewport;
    ofRectangle _mask_viewport;
    ofRectangle _hand_viewport;
    ofRectangle _centers_viewport;
    ofRectangle _grasp_viewport;
    
    int _train_tile_rows;
    int _train_tile_cols;
    std::vector<ofImage> _train_imgs_of;
    std::vector<ofImage> _train_msks_of;
    int _tile_start_index;
    
    ofEasyCam _ptcloud_camera;
    //ofEasyCam _object_camera;
    
    // addjust for better 3d view
    int z_offset;
    
    ofMesh _3d_mesh;
    
    int _calibrate_phase;
    
    int _last_mouse_x;
    int _last_mouse_y;
    
    
    void draw_roi(AppContext *context);
    void draw_ptcloud_centers(AppContext *context, ofRectangle &viewport, const ofVec3f &pt1, const ofVec3f &pt2, const ofVec3f &pt3);
    void draw_object_grasp(AppContext *context, ofRectangle &viewport);
    void draw_training_tiles(AppContext *context);
    
    void transform_viewport(ofRectangle &viewport1, ofRectangle &viewport2, ofRectangle rect1, ofRectangle &rect2);
    
public:
    void setup(AppContext *context);
    void start_calibrate();
    bool calibrate(AppContext *context);
    void update(AppContext *context);
    void draw(AppContext *context);
    
    void go_home();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(AppContext *context, int x, int y );
    void mouseDragged(AppContext *context, int x, int y, int button);
    void mousePressed(AppContext *context, int x, int y, int button);
    void mouseReleased(AppContext *context, int x, int y, int button);
};


#endif
