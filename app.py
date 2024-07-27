# app.py
import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the trained model
clf = joblib.load('clf_model.pkl')

# Load the scaler
sc = joblib.load('scaler.pkl')

# Load the feature selector
with open('feature_selector.pkl', 'rb') as f:
    fs = pickle.load(f)

# Load the dataset to get column names and statistics for user inputs
df = pd.read_csv('breast-cancer.csv')

st.title('Breast Cancer Prediction App')
st.write('This is a simple web app to predict breast cancer.')

# User input
st.sidebar.header('User Input Parameters')



# Interactive Prediction
st.write("## Make Predictions")
def user_input_features():
    features = {}
    # Load the preprocessed dataset to get feature names
    df_features = pd.read_csv('preprocessed_breast_cancer.csv')
    feature_names = df_features.columns[df_features.columns != 'diagnosis']  # Exclude 'diagnosis'
    
    for feature in feature_names:
        min_value = float(df_features[feature].min())
        max_value = float(df_features[feature].max())
        mean_value = float(df_features[feature].mean())
        features[feature] = st.sidebar.slider(feature, min_value, max_value, mean_value)
    
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Load scaler and model
scaler = pickle.load(open('scaler.pkl', 'rb'))
clf = joblib.load('clf_model.pkl')

# Apply model to make predictions
scaled_input = scaler.transform(input_df)
selected_features = fs.transform(scaled_input)
prediction = clf.predict(selected_features)

st.write("### Prediction")
st.write("Benign" if prediction[0] == 0 else "Malignant")

# Save the file, run the Streamlit app
# streamlit run app.py
