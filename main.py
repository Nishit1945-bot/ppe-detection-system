"""
Flask web application for PPE detection with browser camera
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.getenv('MODEL_PATH', 'data/models/best.pt')
model = None

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'ready' if model_exists else 'model_not_found',
        'model_path': MODEL_PATH,
        'model_exists': model_exists
    })

@app.route('/detect', methods=['POST'])
def detect():
    """API endpoint for detection"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    image_data = data.get('image', '')
    
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400
    
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(frame, conf=0.5)[0]
    
    annotated = results.plot()
    
    ret, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    detected_items = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        confidence = float(box.conf[0])
        detected_items.append({
            'class': class_name,
            'confidence': round(confidence, 2)
        })
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}',
        'detections': detected_items
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
