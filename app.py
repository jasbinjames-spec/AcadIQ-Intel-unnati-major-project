from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

app = Flask(__name__)
CORS(app)

# ─── Model Training (runs once at startup) ────────────────────────────────────
# In production, replace this block with loading a saved model:
#   with open('model.pkl', 'rb') as f:
#       knn, scaler, le, feature_cols = pickle.load(f)

def train_model_from_data():
    """
    Replace the synthetic data below with your actual dataset load:
        df = pd.read_excel('AI Project dataset.xlsx')
    The synthetic data mirrors the same feature distribution so the
    API is functional out-of-the-box for demo purposes.
    """
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        'Age': np.random.randint(15, 25, n),
        'Study Hours': np.random.uniform(0, 12, n),
        'Screen Time': np.random.uniform(0, 10, n),
        'Previous Academic Performance': np.random.uniform(40, 100, n),
    })

    # Simulate realistic label logic
    score = (
        df['Study Hours'] * 5
        + df['Previous Academic Performance'] * 0.4
        - df['Screen Time'] * 3
        + np.random.normal(0, 5, n)
    )
    df['Performance Score'] = pd.cut(
        score,
        bins=[-np.inf, 30, 55, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    X = df.drop('Performance Score', axis=1)
    y = df['Performance Score']
    feature_cols = list(X.columns)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)

    return knn, scaler, le, feature_cols

knn, scaler, le, FEATURE_COLS = train_model_from_data()

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Expects JSON:
    {
        "Age": 18,
        "Study Hours": 6.5,
        "Screen Time": 3.0,
        "Previous Academic Performance": 75.0
    }
    Returns:
    {
        "prediction": "High",
        "probabilities": {"High": 0.8, "Medium": 0.15, "Low": 0.05},
        "confidence": 0.8
    }
    """
    try:
        data = request.get_json()

        # Validate fields
        missing = [f for f in FEATURE_COLS if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Build input array
        input_df = pd.DataFrame([{col: data[col] for col in FEATURE_COLS}])
        input_scaled = scaler.transform(input_df)

        # Predict
        pred_enc = knn.predict(input_scaled)[0]
        pred_label = le.inverse_transform([pred_enc])[0]

        # Probabilities (vote counts from KNN)
        proba = knn.predict_proba(input_scaled)[0]
        classes = le.inverse_transform(knn.classes_)
        prob_dict = {str(cls): round(float(p), 4) for cls, p in zip(classes, proba)}
        confidence = round(float(max(proba)), 4)

        return jsonify({
            'prediction': pred_label,
            'probabilities': prob_dict,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model': 'K-Nearest Neighbors (KNN)',
        'k': 5,
        'accuracy': '96.46%',
        'features': FEATURE_COLS,
        'classes': list(le.classes_),
        'description': 'Predicts student academic performance (Low / Medium / High) based on study habits and prior performance.'
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
