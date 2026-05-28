from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
import time
import os
import json
import base64
from db_manager import init_db, insert_record, get_recent_records, get_summary_stats
from yolo_detector import YoloDetector
from ml_classifier import MlClassifier
from hardware_control import HardwareController

app = Flask(__name__)

# Initialize subsystems
init_db()
detector = YoloDetector()
ml_clf = MlClassifier()
hardware = HardwareController()

# Camera instance (shared thread-safe)
camera = None
camera_lock = threading.Lock()

# Mapping from YOLO classes to nominal dimensions (mm) for DB recording
NOMINAL_SIZES = {
    'M3': 3.0,
    'M4': 4.0,
    'M5': 5.0,
    'M6': 6.0
}

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            # Open USB Webcam (typically index 0)
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Warning: Could not open USB Webcam. A simulated camera feed will be used.")
        return camera

def generate_video_stream():
    """Generates a live video stream from the webcam."""
    cap = get_camera()
    while True:
        if cap is not None and cap.isOpened():
            success, frame = cap.read()
            if not success:
                # Fallback to simulated feed if camera read fails
                frame = create_simulated_frame()
        else:
            frame = create_simulated_frame()
            time.sleep(0.05)
            
        # Encode as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def create_simulated_frame():
    """Creates a gray test screen when no physical camera is found."""
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 200
    cv2.putText(img, "SIMULATED FEED (No USB Camera)", (80, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route that returns the live MJPEG stream."""
    return Response(generate_video_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/inspect', methods=['POST'])
def inspect():
    """Performs the hybrid inspection sequence: YOLOv8n object detection -> Machine Learning QC decision."""
    # Check if we should inject a scratch for testing (always update environment first)
    req_data = request.get_json(silent=True) or {}
    if req_data.get('mock_scratch'):
        os.environ['MOCK_SCRATCH'] = '1'
    else:
        os.environ['MOCK_SCRATCH'] = '0'

    cap = get_camera()
    frame = None
    
    # 1. Capture image
    if cap is not None and cap.isOpened():
        # Flush buffer to get latest frame
        for _ in range(5):
            cap.grab()
        success, frame = cap.read()
        
    if frame is None:
        # If camera not working, generate a mock sample for testing
        print("Using simulated sample image for inspection.")
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
            
    # Save a copy of the raw capture
    raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latest_capture.jpg')
    cv2.imwrite(raw_path, frame)
    
    # 2. Run YOLOv8n Inference to detect objects
    detect_res = detector.detect(frame)
    
    scratch_detected = detect_res['scratch_detected']
    scratches = detect_res['scratches']
    holes = detect_res['holes']
    final_annotated_img = detect_res['annotated_image']
    
    # 3. Extract Features for the Machine Learning Classifier
    scratch_count = len(scratches)
    m3_count = len([h for h in holes if h['class_name'] == 'M3'])
    m4_count = len([h for h in holes if h['class_name'] == 'M4'])
    m5_count = len([h for h in holes if h['class_name'] == 'M5'])
    m6_count = len([h for h in holes if h['class_name'] == 'M6'])
    
    # Calculate average confidence across all detections
    confidences = [h['confidence'] for h in holes] + [s['confidence'] for s in scratches]
    avg_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
    total_objects = len(holes) + scratch_count
    
    # Assemble feature vector dictionary
    ml_features = {
        'scratch_count': scratch_count,
        'm3_count': m3_count,
        'm4_count': m4_count,
        'm5_count': m5_count,
        'm6_count': m6_count,
        'avg_confidence': avg_confidence,
        'total_objects': total_objects
    }
    
    # 4. Predict final QC status using the Decision Tree Machine Learning Classifier
    status, pred_label = ml_clf.predict(ml_features)
    
    fail_reason = None
    if status == 'FAIL':
        if scratch_count > 0:
            fail_reason = "ML Classification: Scratch defect detected."
        elif total_objects == 0:
            fail_reason = "ML Classification: No objects/holes detected."
        else:
            fail_reason = f"ML Classification: Quality checks failed (Hole pattern incorrect)."
            
    # Extract hole classifications and map to nominal sizes for DB
    hole_sizes = []
    classifications = []
    for h in holes:
        cls_name = h['class_name']
        classifications.append(cls_name)
        # Record nominal mm size based on classification label
        hole_sizes.append(NOMINAL_SIZES.get(cls_name, 0.0))
        
    # 5. Trigger Hardware GPIO Signals (Non-blocking)
    if status == "PASS":
        threading.Thread(target=hardware.signal_pass).start()
    else:
        threading.Thread(target=hardware.signal_fail).start()
        
    # 6. Save Inspection Result to SQLite
    insert_record(
        status=status,
        scratch_detected=scratch_detected,
        hole_count=len(holes),
        hole_sizes=hole_sizes,
        classification=classifications,
        fail_reason=fail_reason
    )
    
    # 7. Encode result image for web UI display
    _, buffer = cv2.imencode('.jpg', final_annotated_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    # 8. Get updated stats
    stats = get_summary_stats()
    
    response = {
        'status': status,
        'scratch_detected': scratch_detected,
        'hole_count': len(holes),
        'hole_sizes': hole_sizes,
        'classifications': classifications,
        'fail_reason': fail_reason,
        'image': f"data:image/jpeg;base64,{encoded_image}",
        'stats': stats,
        'holes_details': [
            {'name': h['class_name'], 'box': h['box'], 'confidence': round(h['confidence'], 2)}
            for h in holes
        ],
        'ml_features': ml_features
    }
    
    return jsonify(response)

@app.route('/history', methods=['GET'])
def history():
    records = get_recent_records(15)
    return jsonify(records)

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(get_summary_stats())

if __name__ == '__main__':
    # On PYNQ-Z2, we run on port 5000 (accessible on local network)
    # Allow connection from external devices (host=0.0.0.0)
    app.run(host='0.0.0.0', port=5000, debug=False)
