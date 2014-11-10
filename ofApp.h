#pragma once

#include "ofMain.h"
#include "ofxKinect.h"
#include "ofxCv.h"

#include "app_context.h"
#include "mask_generator.h"
#include "hand_detector.h"
#include "grasp_classifier.h"
#include "torso_com.h"
#include "object_detector.h"
#include "ui.h"
#include "demo_ui.h"

#include <string>

class ofApp : public ofBaseApp{

	public:
    
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
    
    ofxKinect kinect;

    MaskGenerator _mask_generator;
    HandDetectorWrapper _hand_detector;
    GraspClassifier _grasp_classifier;
    TorsoCom _torso_com;
    
    ObjectDetector _object_detector;
    
    
    UI _ui;
    DemoUI _demo_ui;
    
    AppContext *_context;
    
    ofTrueTypeFont	_verdana;

    bool _debug_mode;
    
    std::string _carlibration_file;
    
    bool _force_restart;
    
    int training_cnt;
    int sample_cnt;
    int save_cnt;
    
};
