# 🎓 Student Academic Performance Predictor

A machine learning web application that predicts a student's academic performance
level — **Low**, **Medium**, or **High** — based on lifestyle and academic habits,
powered by a K-Nearest Neighbors (KNN) classifier and served via a Flask REST API.

---

## 📸 Demo

> <img width="1901" height="666" alt="image" src="https://github.com/user-attachments/assets/be63c8fe-4b9f-40f1-bbfc-7b1bff938890" />
<img width="1902" height="711" alt="image" src="https://github.com/user-attachments/assets/52efa8d7-a1b5-4295-a237-38b3f47ac0f0" />



---

## 📌 Features

- 🔍 Predicts student performance: **Low / Medium / High**
- 📊 Trained on real student dataset with **96.46% accuracy**
- ⚡ Fast REST API built with Flask
- 🌐 Clean HTML/CSS frontend interface
- 🔄 Easily retrain the model on new data

---

## 🛠️ Tech Stack

| Layer | Technology |
|------------|--------------------------------------|
| Language | Python 3 |
| ML Model | K-Nearest Neighbors (Scikit-learn) |
| Backend | Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript |
| Data | Pandas, NumPy, OpenPyXL |
| Server | Gunicorn |

---

## 📂 Project Structure

​```

├── app.py                  # Flask backend & API routes

├── save_model.py           # Script to train & save model

├── model.pkl               # Trained model (binary)

├── index.html              # Frontend UI

├── AI_Project_dataset.xlsx # Student dataset

├── requirements.txt        # Python dependencies

└── README.md               # Project documentation

​```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/jasbinjames-spec/AcadIQ-Intel-unnati-major-project.git
cd AcadIQ-Intel-unnati-major-project

```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train and save the model
```bash
python save_model.py
```

### 4. Run the Flask app
```bash
python app.py
```

### 5. Open in your browser
​```
http://localhost:5000
​```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|-----------------|-------------------------------|
| POST | `/api/predict` | Predict student performance |
| GET | `/api/model-info` | Get model details & accuracy |
| GET | `/api/health` | Check if API is running |

### Sample Request — `/api/predict`

```json
{
  "Age": 18,
  "Study Hours": 6.5,
  "Screen Time": 2.0,
  "Previous Academic Performance": 78.0
}
```

### Sample Response

```json
{
  "prediction": "High",
  "probabilities": {
    "High": 0.8,
    "Medium": 0.15,
    "Low": 0.05
  },
  "confidence": 0.8
}
```

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Algorithm | K-Nearest Neighbors (KNN) |
| K value | 5 |
| Accuracy | 96.46% |
| Input Features | Age, Study Hours, Screen Time, Previous Academic Performance |
| Output Classes | Low, Medium, High |

---

## 👤 Authors

Jasbin James
Justin K. Samuel
Kavya Singh
Kripa Mariam Jhon
