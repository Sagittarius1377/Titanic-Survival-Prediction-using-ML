# 🚢 Titanic Survival Prediction using Machine Learning

## 🌐 Live Demo

👉 https://titanic-survival-prediction-using-ml-esoktappw.streamlit.app/

---

## 📌 Project Overview

This project predicts whether a passenger would survive the Titanic disaster using Machine Learning.

It demonstrates a **complete end-to-end ML pipeline**, including:

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Web application deployment
* Docker containerization

---

## 🎯 Key Features

* 🔍 Predict survival based on passenger inputs
* 📊 Interactive Streamlit UI
* 🧠 Random Forest ML model (82% accuracy)
* 📈 Visualizations:

  * Feature Importance
  * Confusion Matrix
  * Survival Heatmap
* ✅ Input validation for realistic predictions
* ⚙️ Auto-calculated features (`IsAlone`)
* 🌐 Live deployment on Streamlit Cloud
* 🐳 Dockerized application (available on Docker Hub)

---

## 🧠 Machine Learning Workflow

### 🔹 Data Preprocessing

* Handled missing values
* Converted categorical data using Label Encoding
* Selected relevant features

### 🔹 Feature Engineering

* `FamilySize = SibSp + Parch + 1`
* `IsAlone = 1 if FamilySize == 1 else 0`

### 🔹 Models Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest ✅ (Final Model)

### 🔹 Model Performance

| Model               | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| Logistic Regression | 79%      | 77%       | 71%    |
| Random Forest       | 82%      | 79%       | 77%    |
| KNN                 | 82%      | 79%       | 77%    |

👉 Random Forest selected based on best performance.

---

## 📊 Visualizations

### 🔹 Feature Importance

Shows which features influence predictions the most.

### 🔹 Confusion Matrix

Helps understand model errors (False Positives & False Negatives).

### 🔹 Survival Heatmap

Shows survival trends across Passenger Class and Gender.

---

## 🖥️ Streamlit Application

### 🎯 User Inputs

* Passenger Class (1, 2, 3)
* Sex
* Age (0–80)
* Fare (0–512)
* Family Size (1–11)
* Embarked (C, Q, S)

### ⚙️ System Logic

* `IsAlone` automatically calculated
* Input validation prevents invalid data
* Prediction displayed with confidence score

---

## 🐳 Docker Support

### 🔹 Docker Image

Available on Docker Hub:

👉 `sagittarius1377/titanic-app:v1`

---

### 🔹 Run using Docker

```bash
docker pull sagittarius1377/titanic-app:v1
docker run -p 8501:8501 sagittarius1377/titanic-app:v1
```

---

## 🚀 Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/Sagittarius1377/Titanic-Survival-Prediction-using-ML.git
cd Titanic-Survival-Prediction-using-ML
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run App

```bash
streamlit run frontend.py
```

---

## 📁 Project Structure

```
Titanic-Survival-Prediction/
│
├── frontend.py          # Streamlit UI
├── model.pkl            # Trained model
├── requirements.txt     # Dependencies
├── runtime.txt          # Python version control
├── src.ipynb            # Model training
├── titanic.csv          # Dataset
├── Dockerfile           # Docker config
├── .dockerignore
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* Docker

---

## 💡 Key Insights

* Females had higher survival rates
* First-class passengers had better chances
* Fare positively influenced survival probability
* Smaller families had higher survival rates

---

## 👩‍💻 Author

**Bhumika Yadav**

---