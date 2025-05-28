import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("RANDOM_FOREST_WINE_CLASSIFIER.pkl")

# App Title and Description
st.title("🍷 Wine Quality Prediction App")
st.markdown(
    """
    This app predicts the **quality of wine** based on its physicochemical properties.  
    Simply adjust the sliders on the left to input wine properties, and click the **Predict** button to see the quality category!  
    """
)

# Sidebar for user input
st.sidebar.header("Input Wine Properties")
def user_input_features():
    fixed_acidity = st.sidebar.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, 7.0)
    volatile_acidity = st.sidebar.slider("Volatile Acidity (g/dm³)", 0.1, 2.0, 0.5)
    citric_acid = st.sidebar.slider("Citric Acid (g/dm³)", 0.0, 1.0, 0.3)
    residual_sugar = st.sidebar.slider("Residual Sugar (g/dm³)", 0.0, 16.0, 5.0)
    chlorides = st.sidebar.slider("Chlorides (g/dm³)", 0.01, 0.1, 0.04)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide (mg/L)", 1, 72, 15)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide (mg/L)", 6, 289, 46)
    density = st.sidebar.slider("Density (g/cm³)", 0.990, 1.004, 0.995)
    pH = st.sidebar.slider("pH", 2.8, 4.0, 3.3)
    sulphates = st.sidebar.slider("Sulphates (g/dm³)", 0.2, 2.0, 0.6)
    alcohol = st.sidebar.slider("Alcohol (% by volume)", 8.0, 15.0, 10.5)

    # Combine all inputs into a DataFrame
    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }
    return pd.DataFrame(data, index=[0])

# Get user inputs
input_df = user_input_features()

# Main Panel: Display Inputs
st.subheader("Your Input Properties")
st.dataframe(input_df)

# Predict Button
if st.button("Predict Wine Quality"):
    # Prediction
    prediction = model.predict(input_df)
    quality_map = {0: "Low", 1: "Medium", 2: "High"}
    predicted_quality = quality_map[prediction[0]]

    # Fun messages based on prediction
    jokes = {
        "Low": "🔴 Oof, this wine might not make the cut at a fancy dinner. Maybe it's better as a cooking wine?",
        "Medium": "🟡 Not bad! This wine might make it to the second round of wine tasting competitions.",
        "High": "🟢 Excellent choice! This wine could impress even the pickiest sommelier."
    }

    # Display Prediction with Fun Sentence
    st.subheader("Prediction Result")
    if predicted_quality == "Low":
        st.error(f"🔴 The wine quality is predicted to be: **{predicted_quality}**")
    elif predicted_quality == "Medium":
        st.warning(f"🟡 The wine quality is predicted to be: **{predicted_quality}**")
    else:
        st.success(f"🟢 The wine quality is predicted to be: **{predicted_quality}**")
    
    # Display fun sentence
    st.markdown(f"**Fun Fact:** {jokes[predicted_quality]}")
