import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

def calculate_angle(v1, v2):
    # calculate angle between two 3d vectors in radians
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_hand_features(pts):
    # extracts 130 spatial features optimized for asl letter accuracy
    pts = np.array(pts)
    wrist = pts[0]
    
    # center landmarks relative to wrist
    rel_pts = pts - wrist
    
    # scale using palm size (wrist to middle mcp) for scale stability across postures
    palm_scale = np.linalg.norm(rel_pts[9])
    if palm_scale == 0: 
        palm_scale = 1e-6
    norm_pts = rel_pts / palm_scale
    
    feats = norm_pts.flatten().tolist()
    
    # distances from wrist to all points
    for i in range(1, 21):
        feats.append(float(np.linalg.norm(norm_pts[i])))
        
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]

    # tip to tip distances (helps with letters like o, c, e, fist shapes)
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            feats.append(float(np.linalg.norm(norm_pts[tips[i]] - norm_pts[tips[j]])))
            
    # tip to mcp distances
    for tip in tips:
        for mcp in mcps:
            feats.append(float(np.linalg.norm(norm_pts[tip] - norm_pts[mcp])))

    finger_joints = [
        [0, 1, 2], [1, 2, 3], [2, 3, 4],       # thumb
        [0, 5, 6], [5, 6, 7], [6, 7, 8],       # index
        [0, 9, 10], [9, 10, 11], [10, 11, 12],  # middle
        [0, 13, 14], [13, 14, 15], [15, 16, 17],# ring
        [0, 17, 18], [17, 18, 19], [18, 19, 20] # pinky
    ]
    for p1, p2, p3 in finger_joints:
        v1 = norm_pts[p1] - norm_pts[p2]
        v2 = norm_pts[p3] - norm_pts[p2]
        feats.append(calculate_angle(v1, v2))

    # thumb tip to finger joint distances (distinguishes m, n, t, a, s)
    for joint in [6, 7, 10, 11, 14, 15]:
        feats.append(float(np.linalg.norm(norm_pts[4] - norm_pts[joint])))

    finger_chains = [
        [1, 4],   # thumb
        [5, 8],   # index
        [9, 12],  # middle
        [13, 16], # ring
        [17, 20]  # pinky
    ]
    for i in range(len(finger_chains) - 1):
        v1 = norm_pts[finger_chains[i][1]] - norm_pts[finger_chains[i][0]]
        v2 = norm_pts[finger_chains[i+1][1]] - norm_pts[finger_chains[i+1][0]]
        feats.append(calculate_angle(v1, v2))

    return feats

def augment_data(x_data, y_data, noise_level=0.015, copies=1):
    x_augmented = [x_data]
    y_augmented = [y_data]
    for _ in range(copies):
        noise = np.random.normal(0, noise_level, x_data.shape)
        x_augmented.append(x_data + noise)
        y_augmented.append(y_data)
    return np.vstack(x_augmented), np.hstack(y_augmented)

def main():
    print("loading csv dataset...")
    df = pd.read_csv('asl_landmarks.csv')
    
    if 'label' in df.columns:
        y = df['label'].values
        X_raw = df.drop(columns=['label']).values
    else:
        y = df.iloc[:, 0].values
        X_raw = df.iloc[:, 1:].values

    # process landmark coordinates into engineered features
    if X_raw.shape[1] == 63:
        print("transforming landmark coordinates into high-accuracy engineered features...")
        X_list = []
        for row in X_raw:
            pts = row.reshape(21, 3)
            feats = extract_hand_features(pts)
            X_list.append(feats)
        X = np.array(X_list)
    else:
        print(f"dataset already contains {X_raw.shape[1]} features.")
        X = X_raw

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # augment training split to prevent overfitting to still images
    X_train, y_train = augment_data(X_train, y_train, noise_level=0.01, copies=1)

    print("training optimized classifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=12,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"model validation accuracy: {acc * 100:.2f}%")

    joblib.dump(clf, 'asl_model.pkl')
    print("saved updated model to asl_model.pkl successfully.")

if __name__ == "__main__":
    main()