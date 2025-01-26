import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Model Loading Functions
def load_cnn_model(model_file_cnn):
    return tf.keras.models.load_model(model_file_cnn)

def load_random_forest(model_file):
    return joblib.load(model_file)

def load_svm_model(model_file):
    return joblib.load(model_file)

# Image Preprocessing
def preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction Functions
def predict_cnn(model, image_array):
    prediction = model.predict(image_array)
    return 'Fraud' if prediction[0][0] > 0.5 else 'Non-Fraud'

def predict_random_forest(model, image_array):
    flattened_image = image_array.reshape(1, -1)
    prediction = model.predict(flattened_image)
    return 'Fraud' if prediction[0] == 1 else 'Non-Fraud'

def predict_svm(model, image_array):
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
        ['Convolutional Neural Network', 'Random Forest', 'Support Vector Machine']
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
            if model_type == 'Convolutional Neural Network':
                model = load_cnn_model('cnn_fraud_model.h5')
                prediction = predict_cnn(model, processed_image)
            
            elif model_type == 'Random Forest':
                model = load_random_forest('random_forest_model.pkl')
                prediction = predict_random_forest(model, processed_image)
            
            elif model_type == 'Support Vector Machine':
                model = load_svm_model('svm_model.pkl')
                prediction = predict_svm(model, processed_image)
            
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