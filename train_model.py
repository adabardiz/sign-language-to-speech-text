import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

def extract_hand_features(pts):
    """
    extracts 126 engineered spatial features from 21 hand landmarks
    """
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
        [1, 2, 3, 4],     # thumb
        [5, 6, 7, 8],     # index
        [9, 10, 11, 12],  # middle
        [13, 14, 15, 16], # ring
        [17, 18, 19, 20]  # pinky
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

    v_index = norm_pts[8] - norm_pts[5]
    v_middle = norm_pts[12] - norm_pts[9]
    norm_idx = np.linalg.norm(v_index)
    norm_mid = np.linalg.norm(v_middle)
    
    if norm_idx > 0 and norm_mid > 0:
        ru_angle = np.dot(v_index, v_middle) / (norm_idx * norm_mid)
        feats.append(float(ru_angle))
    else:
        feats.append(0.0)

    feats.append(float(np.linalg.norm(norm_pts[4] - norm_pts[6])))
    feats.append(float(np.linalg.norm(norm_pts[4] - norm_pts[10])))

    return feats

def main():
    print("Loading CSV dataset...")
    df = pd.read_csv('asl_landmarks.csv')
    
    if 'label' in df.columns:
        y = df['label'].values
        X_raw = df.drop(columns=['label']).values
    else:
        y = df.iloc[:, 0].values
        X_raw = df.iloc[:, 1:].values

    #transform 63 raw coordinate features into 126 engineered features
    if X_raw.shape[1] == 63:
        print("Transforming 63 landmark coordinates into 126 engineered features...")
        X_list = []
        for row in X_raw:
            pts = row.reshape(21, 3)
            feats = extract_hand_features(pts)
            X_list.append(feats)
        X = np.array(X_list)
    else:
        print(f"Dataset already contains {X_raw.shape[1]} features.")
        X = X_raw

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training classifier...")
    clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model validation accuracy: {acc * 100:.2f}%")

    joblib.dump(clf, 'asl_model.pkl')
    print("Saved updated model to asl_model.pkl successfully.")

if __name__ == "__main__":
    main()