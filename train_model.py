import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def extract_hand_features(pts):
    pts = np.array(pts)
    wrist = pts[0]
    rel_pts = pts - wrist
    
    scale = np.linalg.norm(rel_pts[9])
    if scale == 0: 
        scale = 1e-6
    norm_pts = rel_pts / scale
    
    feats = norm_pts.flatten().tolist()
    
    for i in range(1, 21):
        feats.append(float(np.linalg.norm(norm_pts[i])))
        
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            feats.append(float(np.linalg.norm(norm_pts[tips[i]] - norm_pts[tips[j]])))
            
    mcps = [2, 5, 9, 13, 17]
    for tip in tips:
        for mcp in mcps:
            feats.append(float(np.linalg.norm(norm_pts[tip] - norm_pts[mcp])))

    finger_chains = [
        [1, 2, 3, 4],     #thumb
        [5, 6, 7, 8],     #index
        [9, 10, 11, 12],  #middle
        [13, 14, 15, 16], #ring
        [17, 18, 19, 20]  #pinky
    ]
    for chain in finger_chains:
        v1 = norm_pts[chain[1]] - norm_pts[chain[0]]
        v2 = norm_pts[chain[3]] - norm_pts[chain[2]]
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 > 0 and norm_v2 > 0:
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            feats.append(float(np.clip(cos_angle, -1.0, 1.0)))
        else:
            feats.append(0.0)

    return feats

def augment_and_extract(data_frame, num_copies=4, noise_level=0.015):
    """
    takes raw landmarks creates slightly noisy clones for training data,
    and extracts advanced geometric features.
    """
    X_raw = data_frame.drop(columns=['label']).values
    y_raw = data_frame['label'].values
    
    features_list = []
    labels_list = []
    
    for i, row in enumerate(X_raw):
        pts = row.reshape(21, 3)
        
        features_list.append(extract_hand_features(pts))
        labels_list.append(y_raw[i])
        
        #generate noisy clones only active when num_copies> 0
        for _ in range(num_copies):
            noise = np.random.normal(0, noise_level, pts.shape)
            noisy_pts = pts + noise
            features_list.append(extract_hand_features(noisy_pts))
            labels_list.append(y_raw[i])
            
    return np.array(features_list), np.array(labels_list)

print("loading dataset...")
data = pd.read_csv('asl_landmarks.csv')

#split raw dataset FIRST 
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

#creates 4 noisy clones per row for only the training set
print("augmenting training set...")
X_train, y_train = augment_and_extract(train_df, num_copies=4)

#pure clean evaluation
print("extracting features for clean test set...")
X_test, y_test = augment_and_extract(test_df, num_copies=0)

print(f"Training on {len(X_train)} augmented samples | Testing on {len(X_test)} clean samples.")

print("training upgraded histgradientboosting classifier...")
clf = HistGradientBoostingClassifier(
    random_state=42, 
    max_iter=500, 
    learning_rate=0.08, 
    l2_regularization=0.1
)
clf.fit(X_train, y_train)

print("evaluating model against clean test data...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\ntrue real world accuracy: {acc * 100:.2f}% ---\n")

print("detailed letter report")
print(classification_report(y_test, y_pred))

#save confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('asl letter confusion matrix (look 4 dark red blocks off the center line)', fontsize=14)
plt.ylabel('actual letter', fontsize=12)
plt.xlabel('predicted Letter by ai', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("saved diagnostic heatmap to 'confusion_matrix.png'")

joblib.dump(clf, 'asl_model.pkl')
print("saved upgraded model to 'asl_model.pkl'!")