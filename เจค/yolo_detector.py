import cv2
import numpy as np
import os

class YoloDetector:
    def __init__(self, model_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """Initializes the multi-class YOLOv8 Detector for scratches and holes.
        
        Args:
            model_path (str): Path to YOLOv8 model (.pt or .onnx)
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.backend = None  # 'ultralytics', 'opencv_onnx', or 'mock'
        self.model = None
        
        # Configurable Class Map (Users can adjust indexes here)
        self.class_map = {
            0: 'scratch',
            1: 'M3',
            2: 'M4',
            3: 'M5',
            4: 'M6'
        }
        
        # Color mapping for visualization (BGR format)
        self.colors = {
            'scratch': (0, 0, 255),    # Red
            'M3': (255, 0, 0),         # Blue
            'M4': (139, 92, 246),      # Purple
            'M5': (236, 72, 153),      # Pink
            'M6': (245, 158, 11),      # Amber/Yellow
            'default': (0, 255, 0)      # Green for others
        }
        
        if model_path is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            pt_path = os.path.join(models_dir, 'yolov8n.pt')
            onnx_path = os.path.join(models_dir, 'yolov8n.onnx')
            
            if os.path.exists(onnx_path):
                model_path = onnx_path
            elif os.path.exists(pt_path):
                model_path = pt_path
            else:
                model_path = onnx_path
                
        self.model_path = model_path

    def load_model(self):
        """Loads the YOLOv8 model."""
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file not found at {self.model_path}")
            print("Detector will run in Multi-Class Simulation (Mock) Mode.")
            self.backend = 'mock'
            return False
            
        ext = os.path.splitext(self.model_path)[1].lower()
        
        if ext == '.pt':
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.backend = 'ultralytics'
                print(f"Loaded PyTorch model using Ultralytics from: {self.model_path}")
                return True
            except ImportError:
                print("Ultralytics library not installed. Cannot load .pt model.")
                self.backend = 'mock'
                return False
        elif ext == '.onnx':
            try:
                self.model = cv2.dnn.readNetFromONNX(self.model_path)
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = 'opencv_onnx'
                print(f"Loaded ONNX model using OpenCV DNN from: {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading ONNX via OpenCV DNN: {e}")
                self.backend = 'mock'
                return False
        else:
            self.backend = 'mock'
            return False

    def detect(self, image):
        """Runs multi-class detection on the input image.
        
        Args:
            image (np.ndarray): Input image in BGR format.
            
        Returns:
            dict: {
                'scratch_detected': bool,
                'scratches': list of dict,
                'holes': list of dict,
                'annotated_image': np.ndarray
            }
        """
        if self.backend is None:
            self.load_model()
            
        if self.backend == 'ultralytics':
            return self._detect_ultralytics(image)
        elif self.backend == 'opencv_onnx':
            return self._detect_opencv_onnx(image)
        else:
            return self._detect_mock(image)

    def _detect_ultralytics(self, image):
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
        
        scratches = []
        holes = []
        scratch_detected = False
        annotated_image = image.copy()
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            
            # Map class ID to name
            class_name = self.class_map.get(cls_id, 'unknown')
            
            det_data = {
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': conf,
                'box': [int(c) for c in xyxy]
            }
            
            if class_name == 'scratch':
                scratch_detected = True
                scratches.append(det_data)
            elif class_name in ['M3', 'M4', 'M5', 'M6']:
                holes.append(det_data)
                
            # Draw visualization
            color = self.colors.get(class_name, self.colors['default'])
            cv2.rectangle(annotated_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_image, label, (int(xyxy[0]), int(xyxy[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        return {
            'scratch_detected': scratch_detected,
            'scratches': scratches,
            'holes': holes,
            'annotated_image': annotated_image
        }

    def _detect_opencv_onnx(self, image):
        h, w, _ = image.shape
        input_size = 640
        blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (input_size, input_size), swapRB=True, crop=False)
        self.model.setInput(blob)
        outputs = self.model.forward()
        
        output = np.squeeze(outputs[0]).T  # Shape: [8400, 4 + num_classes]
        
        boxes = []
        confidences = []
        class_ids = []
        
        x_factor = w / input_size
        y_factor = h / input_size
        
        for row in output:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]
            
            if confidence >= self.conf_threshold:
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
                
                cx, cy, box_w, box_h = row[0], row[1], row[2], row[3]
                left = int((cx - box_w/2) * x_factor)
                top = int((cy - box_h/2) * y_factor)
                width = int(box_w * x_factor)
                height = int(box_h * y_factor)
                
                boxes.append([left, top, width, height])
                
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        
        scratches = []
        holes = []
        scratch_detected = False
        annotated_image = image.copy()
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                left, top, width, height = box
                conf = confidences[i]
                cls_id = class_ids[i]
                
                class_name = self.class_map.get(cls_id, 'unknown')
                
                det_data = {
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'box': [left, top, left + width, top + height]
                }
                
                if class_name == 'scratch':
                    scratch_detected = True
                    scratches.append(det_data)
                elif class_name in ['M3', 'M4', 'M5', 'M6']:
                    holes.append(det_data)
                    
                color = self.colors.get(class_name, self.colors['default'])
                cv2.rectangle(annotated_image, (left, top), (left + width, top + height), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(annotated_image, label, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
        return {
            'scratch_detected': scratch_detected,
            'scratches': scratches,
            'holes': holes,
            'annotated_image': annotated_image
        }

    def _detect_mock(self, image):
        """Simulates multi-class YOLOv8n output for local testing."""
        h, w, _ = image.shape
        annotated_image = image.copy()
        
        scratches = []
        holes = []
        scratch_detected = False
        
        # Simulate scratch if toggle active
        if os.environ.get('MOCK_SCRATCH') == '1':
            scratch_detected = True
            box = [int(w * 0.2), int(h * 0.3), int(w * 0.7), int(h * 0.35)]
            det = {
                'class_id': 0,
                'class_name': 'scratch',
                'confidence': 0.92,
                'box': box
            }
            scratches.append(det)
            color = self.colors['scratch']
            cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(annotated_image, "scratch: 0.92", (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        # Simulate M3, M4, M5 holes
        mock_holes = [
            {'class_id': 1, 'name': 'M3', 'box': [130, 220, 170, 260], 'conf': 0.88},
            {'class_id': 2, 'name': 'M4', 'box': [300, 215, 345, 260], 'conf': 0.94},
            {'class_id': 3, 'name': 'M5', 'box': [455, 210, 505, 260], 'conf': 0.91}
        ]
        
        for mh in mock_holes:
            det = {
                'class_id': mh['class_id'],
                'class_name': mh['name'],
                'confidence': mh['conf'],
                'box': mh['box']
            }
            holes.append(det)
            color = self.colors[mh['name']]
            cv2.rectangle(annotated_image, (mh['box'][0], mh['box'][1]), (mh['box'][2], mh['box'][3]), color, 2)
            cv2.putText(annotated_image, f"{mh['name']}: {mh['conf']:.2f}", (mh['box'][0], mh['box'][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        return {
            'scratch_detected': scratch_detected,
            'scratches': scratches,
            'holes': holes,
            'annotated_image': annotated_image
        }

if __name__ == '__main__':
    # Test multi-class detector
    detector = YoloDetector()
    test_img = np.zeros((480, 640, 3), dtype=np.uint8) + 180
    res = detector.detect(test_img)
    print("Scratch Detected:", res['scratch_detected'])
    print("Holes Count:", len(res['holes']))
    print("Holes Details:", [h['class_name'] for h in res['holes']])
