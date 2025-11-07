"""
Flask web application for PPE detection
"""
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.getenv('MODEL_PATH', 'data/models/best.pt')
CONFIDENCE = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

detector = None

def init_detector():
    """Initialize detector"""
    global detector
    if detector is None and os.path.exists(MODEL_PATH):
        detector = YOLO(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    return detector is not None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """API endpoint to check system status"""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'ready' if model_exists else 'model_not_found',
        'model_path': MODEL_PATH,
        'model_exists': model_exists,
        'message': 'System ready' if model_exists else 'Model not found'
    })

def generate_frames():
    """Generate video frames with detection"""
    if not init_detector():
        return
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Cannot open camera")
        return
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            results = detector(frame, conf=CONFIDENCE)[0]
            annotated = results.plot()
            
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
