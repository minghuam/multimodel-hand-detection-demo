//
//  demo_ui.h
//  HandDemo
//
//  Created by Akanksha Saran on 10/24/14.
//
//

#ifndef __HandDemo__demo_ui__
#define __HandDemo__demo_ui__

#include "ofMain.h"
#include "app_context.h"
#include "ofxCv.h"

class DemoUI{
    
private:
    ofRectangle _title_viewport;
    ofRectangle _hand_viewport;
    ofRectangle _centers_viewport;
    
    ofImage _hand_of;
    
    ofEasyCam _ptcloud_camera;
    
    int _z_offset;
    
    ofMesh _3d_mesh;
    
    
    void draw_ptcloud_centers(AppContext *context, ofRectangle &viewport, const ofVec3f &pt1, const ofVec3f &pt2, const ofVec3f &pt3);

    
public:
    void setup(AppContext *context);
    void update(AppContext *context);
    void draw(AppContext *context);
};


#endif /* defined(__HandDemo__demo_ui__) */
