import json
import os
import numpy as np

# Configurable paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'ml_model.json')

FEATURE_NAMES = [
    'scratch_count',
    'm3_count',
    'm4_count',
    'm5_count',
    'm6_count',
    'avg_confidence',
    'total_objects'
]

def generate_mock_dataset():
    """Generates a representative dataset for training the QC classifier.
    
    Features in order:
    [scratch_count, m3_count, m4_count, m5_count, m6_count, avg_confidence, total_objects]
    """
    X = []
    y = []
    
    # 1. Normal plates (PASS) -> label 1
    # No scratches, at least 1 hole, high confidence
    for _ in range(100):
        m3 = np.random.randint(0, 3)
        m4 = np.random.randint(0, 3)
        m5 = np.random.randint(0, 3)
        m6 = np.random.randint(0, 3)
        if m3 + m4 + m5 + m6 == 0:
            m4 = 1 # Make sure there's at least one hole
        total = m3 + m4 + m5 + m6
        avg_conf = np.random.uniform(0.80, 0.98)
        X.append([0, m3, m4, m5, m6, avg_conf, total])
        y.append(1)
        
    # 2. Defective plates with scratches (FAIL) -> label 0
    for _ in range(50):
        scratch = np.random.randint(1, 4)
        m3 = np.random.randint(0, 3)
        m4 = np.random.randint(0, 3)
        total = scratch + m3 + m4
        avg_conf = np.random.uniform(0.60, 0.95)
        X.append([scratch, m3, m4, 0, 0, avg_conf, total])
        y.append(0)
        
    # 3. Defective plates with missing holes (FAIL) -> label 0
    # No scratches but 0 holes
    for _ in range(50):
        X.append([0, 0, 0, 0, 0, 0.0, 0])
        y.append(0)
        
    return np.array(X), np.array(y)

def serialize_tree(decision_tree, feature_names):
    """Recursively serializes the scikit-learn Decision Tree to a dict."""
    tree = decision_tree.tree_
    
    def recurse(node):
        if tree.feature[node] != -2:  # Internal node
            feature_idx = tree.feature[node]
            feature_name = feature_names[feature_idx]
            threshold = float(tree.threshold[node])
            return {
                'type': 'split',
                'feature': feature_name,
                'threshold': threshold,
                'left': recurse(tree.children_left[node]),   # If feature <= threshold
                'right': recurse(tree.children_right[node])   # If feature > threshold
            }
        else:  # Leaf node
            class_values = tree.value[node][0]
            class_label = int(np.argmax(class_values))
            return {
                'type': 'leaf',
                'class': class_label
            }
            
    return recurse(0)

def train_and_export():
    try:
        from sklearn.tree import DecisionTreeClassifier
        print("Scikit-learn detected. Training Decision Tree model...")
        
        X, y = generate_mock_dataset()
        
        # Train decision tree with max_depth to keep the rules simple and interpretable
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X, y)
        
        # Export tree rules to dict and save as JSON
        tree_dict = serialize_tree(clf, FEATURE_NAMES)
        
        with open(MODEL_PATH, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=4)
            
        print(f"Model successfully trained and exported to: {MODEL_PATH}")
        return True
        
    except ImportError:
        print("Scikit-learn is not installed. Exporting a high-quality hand-crafted fallback decision tree JSON...")
        
        # Create a fallback tree that mimics the decision boundaries:
        # If scratch_count > 0.5 (i.e. scratch_count >= 1) -> FAIL (0)
        # Else: If total_objects < 0.5 (i.e. total_objects == 0) -> FAIL (0)
        # Else: PASS (1)
        fallback_tree = {
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
        
        with open(MODEL_PATH, 'w', encoding='utf-8') as f:
            json.dump(fallback_tree, f, indent=4)
            
        print(f"Fallback model successfully exported to: {MODEL_PATH}")
        return False

if __name__ == '__main__':
    train_and_export()
