import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image

# Setting page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Loading the Model
model = load_model('D:\Plant_disease_detection\plant_disease.h5')

# Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Setting Title and Description
st.markdown("<h1 style='text-align: center; color: green;'>Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of the plant leaf to detect the disease</p>", unsafe_allow_html=True)

# Uploading the plant image
st.markdown("<p style='text-align: center;'>Upload an image of the plant leaf:</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    plant_image = st.file_uploader("Choose an image...", type="jpg")

submit = st.button('Predict')

# On predict button click
if submit:
    with st.spinner('Processing...'):
        if plant_image is not None:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Displaying the image
            st.image(opencv_image, channels="BGR", use_column_width=True, caption='Uploaded Leaf Image')
            st.write(opencv_image.shape)
            
            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (256, 256))
            opencv_image.shape = (1, 256, 256, 3)
            
            # Make Prediction
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            
            st.markdown(f"<h2 style='text-align: center;'>This is a {result.split('-')[0]} leaf with {result.split('-')[1]}</h2>", unsafe_allow_html=True)
        else:
            st.error("Please upload an image.")

# Sidebar information
st.sidebar.header("About the App")
st.sidebar.text("This app helps in detecting plant diseases from leaf images. Upload an image and the model will predict the type of disease.")

# Adding custom CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f4f4f4;
    }
    .sidebar .sidebar-content {
        background: #2e7d32;
        color: white;
    }
    h1 {
        color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)
