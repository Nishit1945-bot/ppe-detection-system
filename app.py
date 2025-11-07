"""
Flask web application for PPE detection
"""
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from app.core.detector import PPEDetector
from config.config import config_by_name
from app.utils.logger import setup_logger
import os

app = Flask(__name__)
CORS(app)

config = config_by_name['development']
config.validate()

logger = setup_logger('ppe-webapp', 
                     log_file=str(config.LOG_DIR / 'webapp.log'),
                     level=config.LOG_LEVEL)

detector = None

def init_detector():
    """Initialize PPE detector"""
    global detector
    if detector is None:
        model_path = config.MODEL_PATH
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
        
        detector = PPEDetector(
            model_path=model_path,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            device=config.DEVICE
        )
        logger.info("Detector initialized successfully")
    return True

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """API endpoint to check system status"""
    model_exists = os.path.exists(config.MODEL_PATH)
    return jsonify({
        'status': 'ready' if model_exists else 'model_not_found',
        'model_path': str(config.MODEL_PATH),
        'model_exists': model_exists,
        'device': config.DEVICE,
        'confidence_threshold': config.CONFIDENCE_THRESHOLD
    })

def generate_frames():
    """Generate video frames with PPE detection"""
    if not init_detector():
        logger.error("Failed to initialize detector")
        return
    
    camera = cv2.VideoCapture(config.CAMERA_SOURCE)
    
    if not camera.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Camera opened, starting detection...")
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            results = detector.detect(frame)
            
            annotated_frame = detector.annotate_frame(frame, results)
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        camera.release()
        logger.info("Camera released")

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logger.info(f"Starting PPE Detection Web App on {config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
