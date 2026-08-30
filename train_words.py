import matplotlib
matplotlib.use("Agg")  
import os
import csv
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from train_model import extract_hand_features

FRAMES_PER_SAMPLE = 30

# face landmarks matching collect_words.py
KEY_FACE_INDICES = [
    1,                  # nose tip anchor
    33, 133, 159, 145,  # left eye
    362, 263, 386, 374, # right eye
    70, 63, 105, 66,    # left eyebrow
    300, 293, 334, 296, # right eyebrow
    61, 291, 0, 17, 13, 14, # outer & inner mouth
    78, 308, 82, 312    # lip curves
]

def get_expected_feature_counts():
    dummy_pts = np.zeros((21, 3))
    num_hand_features = len(extract_hand_features(dummy_pts))
    num_face_features = len(KEY_FACE_INDICES) * 3
    num_features_per_frame = num_hand_features + num_face_features
    expected_total_vals = FRAMES_PER_SAMPLE * num_features_per_frame
    return num_features_per_frame, expected_total_vals

def aggregate_sequence(sequence_matrix):
    # flatten 30 frames into an aggregate vector so exact timing doesn't mess up predictions
    seq = np.array(sequence_matrix, dtype=np.float32)
    mean_f = np.mean(seq, axis=0)       # average hand position
    std_f = np.std(seq, axis=0)         # movement variance
    delta_f = seq[-1] - seq[0]          # total displacement from start to end
    max_f = np.max(seq, axis=0)         # peak hand coordinates
    min_f = np.min(seq, axis=0)         # min hand coordinates
    
    return np.hstack([mean_f, std_f, delta_f, max_f, min_f])

def augment_sequence(seq_matrix):
    # generate fake training variations to boost accuracy without re-recording
    augmented = []
    seq = np.array(seq_matrix, dtype=np.float32)
    
    # original raw sequence
    augmented.append(aggregate_sequence(seq))
    
    # subtle hand jitter/noise
    noise = np.random.normal(0, 0.004, seq.shape)
    augmented.append(aggregate_sequence(seq + noise))
    
    # slight scale tweak (simulates hand being closer or farther)
    scale = np.random.uniform(0.95, 1.05)
    augmented.append(aggregate_sequence(seq * scale))

    # speed warping (simulates signing slightly faster or slower)
    indices_fast = np.linspace(0, FRAMES_PER_SAMPLE - 1, FRAMES_PER_SAMPLE, dtype=int)
    shift = np.random.choice([-1, 1], size=FRAMES_PER_SAMPLE)
    indices_warped = np.clip(indices_fast + shift, 0, FRAMES_PER_SAMPLE - 1)
    augmented.append(aggregate_sequence(seq[indices_warped]))

    return augmented

def load_dataset(csv_path="asl_words.csv"):
    if not os.path.exists(csv_path):
        print(f"error: '{csv_path}' not found.")
        return None, None

    labels = []
    features = []
    
    num_features_per_frame, expected_total_vals = get_expected_feature_counts()
    skipped_rows = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row or len(row) <= 1:
                continue
            
            label = row[0].strip().upper()
            try:
                feat_values = [float(val) for val in row[1:]]
            except ValueError:
                continue

            # check row matches exact expected length from collect_words.py
            if len(feat_values) != expected_total_vals:
                skipped_rows += 1
                continue
            
            raw_seq = np.array(feat_values).reshape(FRAMES_PER_SAMPLE, num_features_per_frame)

            # augment and aggregate sequences
            aug_features = augment_sequence(raw_seq)
            for feat_vec in aug_features:
                labels.append(label)
                features.append(feat_vec)

    if skipped_rows > 0:
        print(f"[warning] skipped {skipped_rows} legacy/incompatible rows not matching target feature size ({expected_total_vals}).")

    if not labels:
        print("error: no valid data rows found in csv matching current feature dimensions.")
        return None, None

    print(f"[info] generated {len(features)} total augmented samples from CSV.")
    return np.array(features, dtype=np.float32), np.array(labels)

def main():
    csv_file = "asl_words.csv"
    model_output_path = "asl_word_model.pkl"

    print("--- starting augmented asl word model training ---")
    x, y = load_dataset(csv_file)
    
    if x is None or len(x) == 0:
        return

    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\nclasses & sample counts after augmentation:")
    for cls, cnt in zip(unique_classes, counts):
        print(f" - {cls}: {cnt} samples")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n---> model test accuracy: {acc * 100:.2f}% <---")

    clf.fit(x, y)
    joblib.dump(clf, model_output_path, compress=3)
    print(f"successfully saved updated model to '{model_output_path}'!")

if __name__ == "__main__":
    main()