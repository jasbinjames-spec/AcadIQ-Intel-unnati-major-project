"""
Run this script once to train the model on your real dataset
and save it as model.pkl — then app.py will load it instead
of re-training from scratch on every restart.

Usage:
    python save_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load your real dataset ─────────────────────────────────────────────────
df = pd.read_excel('AI Project dataset.xlsx')   # <── update path if needed

# ── 2. Split features / target ────────────────────────────────────────────────
X = df.drop('Performance Score', axis=1)
y = df['Performance Score']
feature_cols = list(X.columns)

# ── 3. Encode labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# ── 5. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 6. Train KNN ──────────────────────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = knn.predict(X_test_s)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 8. Persist ────────────────────────────────────────────────────────────────
bundle = (knn, scaler, le, feature_cols)
with open('model.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("✅  model.pkl saved — update app.py to load it instead of re-training.")
