import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("Loading word dataset from asl_words.csv...")
    try:
        df = pd.read_csv('asl_words.csv', header=None)
    except FileNotFoundError:
        print("Error: asl_words.csv not found. Run collect_words.py first.")
        return

    y = df.iloc[:, 0].values        
    X = df.iloc[:, 1:].values       # 3,780 features per sample (30 framesx126 features)

    print(f"Dataset shape: {X.shape} ({len(np.unique(y))} unique words)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training word classifier...")
    clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Word Model Validation Accuracy: {acc * 100:.2f}%")

    joblib.dump(clf, 'asl_word_model.pkl')
    print("Saved asl_word_model.pkl successfully!")

if __name__ == "__main__":
    main()