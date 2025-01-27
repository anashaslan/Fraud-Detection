import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Model Loading Functions
def load_logistic_regression(model_file):
    return joblib.load(model_file)

def load_random_forest(model_file):
    return joblib.load(model_file)

def load_decision_tree(model_file):
    return joblib.load(model_file)

# Image Preprocessing
def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess the image to match the input shape expected by the models.
    - Resize the image to the target size.
    - Normalize pixel values to [0, 1].
    - Flatten the image to a 1D array.
    """
    img = Image.open(image)
    img = img.resize(target_size)  # Resize to match the expected input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, -1)  # Flatten to 1D array
    return img_array

# Prediction Functions
def predict_logistic_regression(model, image_array):
    flattened_image = image_array.reshape(1, -1)
    prediction = model.predict(flattened_image)
    return 'Fraud' if prediction[0] == 1 else 'Non-Fraud'

def predict_random_forest(model, image_array):
    flattened_image = image_array.reshape(1, -1)
    prediction = model.predict(flattened_image)
    return 'Fraud' if prediction[0] == 1 else 'Non-Fraud'

def predict_decision_tree(model, image_array):
    flattened_image = image_array.reshape(1, -1)
    prediction = model.predict(flattened_image)
    return 'Fraud' if prediction[0] == 1 else 'Non-Fraud'

# Streamlit App
def main():
    st.title('Insurance Claims Fraud Detection')
    st.write("Upload an image and select a model to predict whether it is Fraud or Non-Fraud.")
    
    # Model Selection
    model_type = st.selectbox(
        'Select Prediction Model',
        ['Logistic Regression', 'Random Forest', 'Decision Tree']
    )
    
    # File Upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        
        try:
            # Preprocess Image
            processed_image = preprocess_image(uploaded_file)
            
            # Load appropriate model
            if model_type == 'Logistic Regression':
                model = load_logistic_regression('logistic_regression_model.pkl')
                prediction = predict_logistic_regression(model, processed_image)
            
            elif model_type == 'Random Forest':
                model = load_random_forest('random_forest_model.pkl')
                prediction = predict_random_forest(model, processed_image)
            
            elif model_type == 'Decision Tree':
                model = load_decision_tree('decision_tree_model.pkl')
                prediction = predict_decision_tree(model, processed_image)
            
            # Display Prediction
            st.subheader('Prediction Result')
            if prediction == 'Fraud':
                st.error(f'Prediction: {prediction} Claim')
            else:
                st.success(f'Prediction: {prediction} Claim')
        
        except Exception as e:
            st.error(f'Error processing image: {e}')

if __name__ == "__main__":
    main()