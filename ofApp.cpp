#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){

    /* kinect setup */
    // enable depth and rgb alignment
    kinect.setRegistration(true);
    // set clipping range
    kinect.setDepthClipping(500, 1200);
    kinect.init();
    kinect.open();
    
    _context = &AppContext::instance();
    _context->kinect = &kinect;
    
    // default roi, entire image region
    ofDirectory dir("./");
    _carlibration_file = _context->data_root + "/calibration.xml";
    cv::FileStorage fs(_carlibration_file.c_str(), FileStorage::READ);
    if(fs.isOpened()){
        fs["hand_roi"] >> _context->hand_roi;
        fs["body_roi"] >> _context->body_roi;
        fs["obj_roi"] >> _context->obj_roi;
    }else{
        _context->body_roi = cv::Rect(0,0,kinect.width,kinect.height);
        _context->hand_roi = cv::Rect(0,0,kinect.width,kinect.height);
        _context->obj_roi = cv::Rect(0,0,kinect.width,kinect.height);
    }
    fs.release();
    
    _mask_generator.setup(_context);
    _hand_detector.setup(_context);
    _grasp_classifier.setup(_context);
    _torso_com.setup(_context);
    _object_detector.setup(_context);
    
    _demo_ui.setup(_context);
    _ui.setup(_context);
    
    _verdana.loadFont("verdana.ttf", 24, true, true);
    _verdana.setSpaceSize(0.6);
    
    
    _debug_mode = false;
    _force_restart = false;
    
    ofShowCursor();
}

//--------------------------------------------------------------
void ofApp::update(){

    if(_force_restart) return;
    
    kinect.update();
    
    if(kinect.isFrameNew()){
        
        _context->is_frame_new = true;
        
        // grab from kinect and update context
        cv::Mat color_mat_raw = cv::Mat(kinect.height, kinect.width, CV_8UC3, kinect.getPixels());
        cv::Mat depth_mat_raw = cv::Mat(kinect.height, kinect.width, CV_8UC1, kinect.getDepthPixels());
        
        cv::Mat color_mat;
        cv::Mat depth_mat;
        
        cv::flip(color_mat_raw, color_mat, 1);
        cv::flip(depth_mat_raw, depth_mat, 1);
        
        //color_mat_raw.copyTo(color_mat);
        //depth_mat_raw.copyTo(depth_mat);
        
        color_mat.copyTo(_context->rgb);
        cvtColor(_context->rgb, _context->bgr, CV_RGB2BGR);
        
        depth_mat.copyTo(_context->depth);
        
        
        _mask_generator.update(_context);

        // PROCESSING
        switch(_context->state){
            case APP_STATE_STARTUP:
                _context->state = APP_STATE_LEARN_BACKGROUND;
                break;
            case APP_STATE_CALIBRATE:
                /* determine ROIs for body(head) and hand */
                if(_ui.calibrate(_context)){
                    // save
                    cv::FileStorage fs(_carlibration_file.c_str(), FileStorage::WRITE);
                    fs << "hand_roi" << _context->hand_roi;
                    fs << "body_roi" << _context->body_roi;
                    fs << "obj_roi" << _context->obj_roi;
                    fs.release();
                    // force restarting...
                    _force_restart = true;
                }
                break;
            case APP_STATE_LEARN_BACKGROUND:
                /* learn background model */
                if(_mask_generator.learn(_context)){
                    _context->state = APP_STATE_RUNNING;
                }
                break;
            case APP_STATE_SAMPLE_HANDMODEL:
                /* sample images and masks for hand models*/
                if((sample_cnt = _hand_detector.sample(_context)) == 0){
                    _context->state = APP_STATE_RUNNING;
                }
                break;
            case APP_STATE_TRAIN_HANDMODEL:
                /* train hand models*/
                if((training_cnt = _hand_detector.train(_context)) == 0){
                    _context->state = APP_STATE_RUNNING;
                }else{
                    _hand_detector.update(_context);
                    _grasp_classifier.update(_context);
                    _torso_com.update(_context);
                }
                break;
            case APP_STATE_SAVING_HANDDETECTION:
                /* save hand detection output as training samples for grasp classification */
                _hand_detector.update(_context);
                if((save_cnt = _hand_detector.save(_context)) == 0){
                    _context->state = APP_STATE_RUNNING;
                }
                break;
            case APP_STATE_RUNNING:
                /* run detection */
                _hand_detector.update(_context);
                _grasp_classifier.update(_context);
                _torso_com.update(_context);
                //_object_detector.update(_context);
                break;
            default:
                break;
        }
        
        // clear up? not neccessary
        color_mat.release();
        depth_mat.release();
        
    }
    
    if(_debug_mode){
        _ui.update(_context);
    }else{
        _demo_ui.update(_context);
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    
    if(_force_restart){
        ofBackground(0);
        _verdana.drawString("please exit and restart.", 300, 300);
        return;
    }
    
    if(!kinect.isConnected()){
        ofBackground(0);
        _verdana.drawString("connecting to kinect...", 300, 300);
        return;
    }
    
    if(_debug_mode){
        _ui.draw(_context);
    }else{
        _demo_ui.draw(_context);
    }
    
    _context->is_frame_new = false;
    
    ofSetColor(255,255,255);
    
    int x = 10;
    int y = 20;
    y += 20;
    ofDrawBitmapString("[d]: toggle debug mode", x, y);
    y += 10;
    ofDrawBitmapString("[b]: learn background model", x, y);
    y += 10;
    ofDrawBitmapString("[s]: sample hand model images", x, y);
    y += 10;
    ofDrawBitmapString("[t]: train hand models", x, y);
    y += 10;
    ofDrawBitmapString("[c]: carlibrate region of interest", x, y);
    y += 10;
    ofDrawBitmapString("[g]: save hand detection output",x,y);
    y += 10;
    ofDrawBitmapString("[f]: toggle fullscreen", x, y);
    
    _verdana.drawString(state_string[_context->state], 300, y);
    if(_context->state == APP_STATE_TRAIN_HANDMODEL){
        _verdana.drawString (ofToString(training_cnt) + "/" + ofToString(31), 800, y);
    }else if(_context->state == APP_STATE_SAMPLE_HANDMODEL){
        _verdana.drawString (ofToString(sample_cnt) + "/" + ofToString(31), 800, y);
    }else if(_context->state == APP_STATE_SAVING_HANDDETECTION){
        _verdana.drawString (ofToString(save_cnt) + "/" + ofToString(1000), 800, y);
    }
    
    char buf[128];
    sprintf(buf, "%.4f,%.4f: ", _context->grasp_scores[0], _context->grasp_scores[1]);
    
    if(_context->state == APP_STATE_RUNNING){
        //_verdana.drawString("grasp type: " + ofToString(_context->grasp_label), 500, y);
        string score_str = buf;
        if(_context->grasp_label==1) _verdana.drawString(score_str + "grasp type: MEDIUM WRAP", 500, y);
        if(_context->grasp_label==2) _verdana.drawString(score_str + "grasp type: PRECISION DISC", 500, y);
        if(_context->grasp_label==0) _verdana.drawString(score_str + "grasp type: NONE", 500, y);
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    switch(key){
        case 'd':
            _debug_mode = !_debug_mode;
            break;
        case 'b':
            _context->state = APP_STATE_LEARN_BACKGROUND;
            _mask_generator.start_learning();
            break;
        case 's':
            _context->state = APP_STATE_SAMPLE_HANDMODEL;
            _hand_detector.start_sampling();
            break;
        case 't':
            _context->state = APP_STATE_TRAIN_HANDMODEL;
            break;
        case 'c':
            _context->state = APP_STATE_CALIBRATE;
            _ui.start_calibrate();
            break;
        case 'g':
            _context->state = APP_STATE_SAVING_HANDDETECTION;
            _hand_detector.start_saving();
            break;
        case 'f':
            ofToggleFullscreen();
            break;
    }
    
    
    
    _ui.keyPressed(key);
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    if(_debug_mode)
    _ui.keyReleased(key);
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    if(_debug_mode)
    _ui.mouseMoved(_context, x,y);
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    if(_debug_mode)
    _ui.mouseDragged(_context, x,y,button);
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    if(_debug_mode)
    _ui.mousePressed(_context, x,y,button);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    if(_debug_mode)
    _ui.mouseReleased(_context, x,y,button);
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
