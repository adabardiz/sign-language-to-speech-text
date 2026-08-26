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

def load_dataset(csv_path="asl_words.csv"):
    if not os.path.exists(csv_path):
        print(f"error: '{csv_path}' not found. run collect_words.py first.")
        return None, None

    labels = []
    features = []

    # inspect the first row to figure out feature dimension automatically
    target_features = None
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and len(row) > 1:
                target_features = len(row) - 1
                break

    if target_features is None:
        print("error: no valid data rows found in csv.")
        return None, None

    print(f"[info] detected target feature vector dimension: {target_features}")

    padded_count = 0
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            
            label = row[0].strip().upper()
            try:
                feat_values = [float(val) for val in row[1:]]
            except ValueError:
                print(f"[warning] skipping corrupt row {row_idx + 1}")
                continue

            # normalize vector sizes across old and new samples
            if len(feat_values) < target_features:
                feat_values = feat_values + [0.0] * (target_features - len(feat_values))
                padded_count += 1
            elif len(feat_values) > target_features:
                feat_values = feat_values[:target_features]

            labels.append(label)
            features.append(feat_values)

    if not labels:
        print("error: no valid data rows found in csv.")
        return None, None

    if padded_count > 0:
        print(f"[note] padded {padded_count} shorter rows to {target_features} features.")

    return np.array(features, dtype=np.float32), np.array(labels)

def main():
    csv_file = "asl_words.csv"
    model_output_path = "asl_word_model.pkl"
    cm_output_path = "confusion_matrix.png"

    print("--- starting asl word model training ---")
    x, y = load_dataset(csv_file)
    
    if x is None or len(x) == 0:
        return

    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"\nloaded {len(x)} total samples across {len(unique_classes)} word classes:")
    for cls, cnt in zip(unique_classes, counts):
        print(f" - {cls}: {cnt} samples")

    can_stratify = all(cnt >= 2 for cnt in counts)
    stratify_param = y if can_stratify else None

    if not can_stratify:
        print("\n[note] some classes have fewer than 2 samples; disabling stratification.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    print(f"\ntraining set size: {len(x_train)} | testing set size: {len(x_test)}")
    print("training random forest classifier...")

    clf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n---> model test accuracy: {acc * 100:.2f}% <---")

    print("\nclassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    try:
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        plt.title("asl word recognition confusion matrix")
        plt.tight_layout()
        plt.savefig(cm_output_path)
        plt.close("all")
        print(f"saved confusion matrix plot to '{cm_output_path}'")
    except Exception as e:
        print("could not generate confusion matrix plot:", e)

    # retrain on full dataset before saving
    print("\nretraining classifier on full dataset for maximum coverage...")
    clf.fit(x, y)

    joblib.dump(clf, model_output_path, compress=3)
    print(f"successfully saved updated model to '{model_output_path}'!")

if __name__ == "__main__":
    main()