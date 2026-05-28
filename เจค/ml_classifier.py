import json
import os

class MlClassifier:
    def __init__(self, model_path=None):
        """Initializes the pure-Python Machine Learning Classifier.
        
        Args:
            model_path (str): Path to exported decision tree JSON model.
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'ml_model.json')
        self.model_path = model_path
        self.model_tree = None
        
    def load_model(self):
        """Loads the decision tree structure from the JSON file."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r', encoding='utf-8') as f:
                    self.model_tree = json.load(f)
                print(f"Loaded Machine Learning model from: {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading ML model JSON: {e}")
                
        print("ML model JSON not found or invalid. Using default fallback Decision Tree rule...")
        # Define default fallback tree structure
        self.model_tree = {
            "type": "split",
            "feature": "scratch_count",
            "threshold": 0.5,
            "left": {
                "type": "split",
                "feature": "total_objects",
                "threshold": 0.5,
                "left": {
                    "type": "leaf",
                    "class": 0
                },
                "right": {
                    "type": "leaf",
                    "class": 1
                }
            },
            "right": {
                "type": "leaf",
                "class": 0
            }
        }
        return False

    def predict(self, features):
        """Predicts the quality class (PASS/FAIL) based on input features.
        
        Args:
            features (dict): Dictionary of features, e.g.:
                {
                    'scratch_count': int,
                    'm3_count': int,
                    'm4_count': int,
                    'm5_count': int,
                    'm6_count': int,
                    'avg_confidence': float,
                    'total_objects': int
                }
                
        Returns:
            tuple: (predicted_class_name (str: 'PASS'/'FAIL'), predicted_label (int: 0/1))
        """
        if self.model_tree is None:
            self.load_model()
            
        current_node = self.model_tree
        
        # Traverse the decision tree
        while current_node.get('type') == 'split':
            feature_name = current_node['feature']
            threshold = current_node['threshold']
            
            # Extract value from features, default to 0 if not present
            val = features.get(feature_name, 0.0)
            
            if val <= threshold:
                current_node = current_node['left']
            else:
                current_node = current_node['right']
                
        # Leaf node reached
        if current_node.get('type') == 'leaf':
            class_label = current_node['class']
            class_name = 'PASS' if class_label == 1 else 'FAIL'
            return class_name, class_label
            
        # Fail safe
        return 'FAIL', 0

if __name__ == '__main__':
    # Test classifier locally
    classifier = MlClassifier()
    classifier.load_model()
    
    # 1. Test normal case
    features_pass = {
        'scratch_count': 0,
        'm3_count': 1,
        'm4_count': 1,
        'm5_count': 0,
        'm6_count': 0,
        'avg_confidence': 0.93,
        'total_objects': 2
    }
    pred_name, pred_label = classifier.predict(features_pass)
    print(f"Test Pass Features: {features_pass} => Prediction: {pred_name} ({pred_label})")
    
    # 2. Test scratch defect case
    features_fail_scratch = {
        'scratch_count': 1,
        'm3_count': 1,
        'm4_count': 0,
        'm5_count': 0,
        'm6_count': 0,
        'avg_confidence': 0.88,
        'total_objects': 2
    }
    pred_name, pred_label = classifier.predict(features_fail_scratch)
    print(f"Test Scratch Features: {features_fail_scratch} => Prediction: {pred_name} ({pred_label})")
    
    # 3. Test empty plate case (missing holes)
    features_fail_empty = {
        'scratch_count': 0,
        'm3_count': 0,
        'm4_count': 0,
        'm5_count': 0,
        'm6_count': 0,
        'avg_confidence': 0.0,
        'total_objects': 0
    }
    pred_name, pred_label = classifier.predict(features_fail_empty)
    print(f"Test Empty Features: {features_fail_empty} => Prediction: {pred_name} ({pred_label})")
