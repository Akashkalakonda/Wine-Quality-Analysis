# Wine-Quality-Analysis

# 🍇 Wine Quality Prediction using Machine Learning  

## 📌 Overview  
This project implements a **Random Forest Classifier** to predict the quality of wine based on its physicochemical properties. The trained model is integrated into an **interactive Streamlit web application**, allowing users to input wine properties and obtain quality predictions in real time.  

## 📊 Dataset  
The dataset used contains various chemical properties of wine, including:  
- **Features:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, pH, sulphates, alcohol content, etc.  
- **Target Variable:** Wine quality categorized into **Low, Medium, and High**.  

## 🔍 Model Development  
- Utilized **Random Forest Classifier** for training.  
- Dataset split into **80% training** and **20% testing**.  
- Model evaluated based on **accuracy, precision, recall, and F1-score**.  
- The trained model is saved as `RANDOM_FOREST_WINE_CLASSIFIER.pkl` using `joblib` for later use.  

### 🏆 Model Performance  
| Metric       | Score  |  
|-------------|--------|  
| Accuracy    | 92.5%  |  
| Precision   | 91.2%  |  
| Recall      | 90.8%  |  
| F1 Score    | 91.0%  |  

## 🎨 Streamlit Web Application  
The **Streamlit app** provides an interactive interface where users can input wine properties using sliders and obtain predictions instantly.  
**Key Features:**  
- User-friendly interface for inputting wine properties.  
- **Real-time predictions** with color-coded quality labels.  
- Fun messages based on the predicted wine quality.  

## 🚀 Installation & Usage  
### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/yourusername/Wine-Quality-Prediction.git  
cd Wine-Quality-Prediction  
```
### 2️⃣ Install Dependencies  
```bash  
pip install -r requirements.txt  
```
### 3️⃣ Run the Streamlit App  
```bash  
streamlit run app.py  
```

## 📂 Project Structure  
```
📁 Wine_Quality_Prediction  
│── WINEDATASET.csv              # Dataset  
│── main.py                      # Model training script  
│── app.py                        # Streamlit web app  
│── RANDOM_FOREST_WINE_CLASSIFIER.pkl  # Saved model  
│── requirements.txt              # Dependencies  
│── README.md                     # Project documentation  
```

