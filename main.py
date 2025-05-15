import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="PlantCare AI - Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #388E3C;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #F1F8E9;
        margin-bottom: 20px;
    }
    .info-text {
        font-size: 1.1rem;
    }
    .highlight {
        color: #1B5E20;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Tensorflow Model Prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
with st.sidebar:
    st.title("PlantCare AI")
    st.markdown("---")
    app_mode = st.radio(
        "Navigation",
        ["Home", "Disease Detection", "About", "Plant Care Tips"]
    )
    st.markdown("---")
    st.info("Developed with ❤️ for plant lovers")

# Class names for prediction
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy']

# Home Page
if app_mode == "Home":
    st.markdown("<h1 class='main-header'>Welcome to PlantCare AI</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image("home_page.jpeg", use_column_width=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Your Plant Health Assistant</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p class='info-text'>PlantCare AI uses advanced machine learning to detect plant diseases from images.</p>
        <p class='info-text'>Protect your garden and crops with early detection and treatment recommendations.</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.button("Get Started", on_click=lambda: st.session_state.update({"app_mode": "Disease Detection"}))
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Detect")
        st.markdown("Upload a photo of your plant and get instant disease diagnosis")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 💡 Learn")
        st.markdown("Understand plant diseases and how to prevent them")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🌱 Grow")
        st.markdown("Apply recommendations to keep your plants healthy")
        st.markdown("</div>", unsafe_allow_html=True)

# Disease Detection Page
elif app_mode == "Disease Detection":
    st.markdown("<h1 class='main-header'>Plant Disease Detection</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>Upload a clear image of a plant leaf to detect diseases. For best results:</p>
    <ul>
        <li>Use well-lit images</li>
        <li>Focus on affected areas</li>
        <li>Avoid shadows or glare</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_image = st.file_uploader("Upload Plant Image:", type=['jpg', 'jpeg', 'png'])
        
        if test_image is not None:
            st.image(test_image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Add a small delay to show the spinner
                    time.sleep(1)
                    try:
                        result_index = model_prediction(test_image)
                        plant_condition = class_name[result_index]
                        
                        # Format the result for better readability
                        plant_type, condition = plant_condition.split('___')
                        condition = condition.replace('_', ' ')
                        
                        st.success(f"Analysis Complete!")
                        
                        with col2:
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            st.markdown("<h3 class='sub-header'>Diagnosis Results</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p class='info-text'><span class='highlight'>Plant Type:</span> {plant_type}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='info-text'><span class='highlight'>Condition:</span> {condition}</p>", unsafe_allow_html=True)
                            
                            # Add treatment recommendations based on condition
                            if "healthy" in plant_condition.lower():
                                st.markdown("<p class='info-text'><span class='highlight'>Status:</span> Your plant appears healthy! 🌿</p>", unsafe_allow_html=True)
                                st.markdown("<p>Continue with regular care and maintenance.</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p class='info-text'><span class='highlight'>Status:</span> Disease detected! ⚠️</p>", unsafe_allow_html=True)
                                st.markdown("<p><span class='highlight'>Recommended Actions:</span></p>", unsafe_allow_html=True)
                                st.markdown("<ul><li>Remove affected leaves</li><li>Improve air circulation</li><li>Consider appropriate fungicide/treatment</li></ul>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during analysis. Please try another image.")
                        st.exception(e)

# About Page
elif app_mode == "About":
    st.markdown("<h1 class='main-header'>About PlantCare AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Our Technology</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>PlantCare AI uses deep learning models trained on thousands of plant images to accurately identify diseases.</p>
    
    <p class='info-text'>Our model can identify 38 different classes of plant diseases across various crops including:</p>
    <ul>
        <li>Apple</li>
        <li>Blueberry</li>
        <li>Cherry</li>
        <li>Corn</li>
        <li>Grape</li>
        <li>Orange</li>
        <li>Peach</li>
        <li>Pepper</li>
        <li>Potato</li>
        <li>Strawberry</li>
        <li>Tomato</li>
        <li>And more!</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Dataset Information</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>This application is powered by a dataset consisting of approximately 87,000 RGB images of healthy and diseased crop leaves categorized into 38 different classes.</p>
    
    <p class='info-text'>The dataset is divided into:</p>
    <ul>
        <li>Training set: 70,295 images</li>
        <li>Validation set: 17,572 images</li>
        <li>Test set: 33 images</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Plant Care Tips Page
elif app_mode == "Plant Care Tips":
    st.markdown("<h1 class='main-header'>Plant Care Tips</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>General Plant Health Tips</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p class='info-text'>Follow these guidelines to maintain healthy plants:</p>
    
    <p><span class='highlight'>1. Proper Watering</span></p>
    <ul>
        <li>Water deeply but infrequently</li>
        <li>Avoid wetting the foliage when possible</li>
        <li>Water early in the day</li>
    </ul>
    
    <p><span class='highlight'>2. Adequate Sunlight</span></p>
    <ul>
        <li>Understand your plant's light requirements</li>
        <li>Rotate plants regularly for even growth</li>
    </ul>
    
    <p><span class='highlight'>3. Regular Inspection</span></p>
    <ul>
        <li>Check plants weekly for signs of pests or disease</li>
        <li>Look under leaves and along stems</li>
        <li>Remove affected parts promptly</li>
    </ul>
    
    <p><span class='highlight'>4. Proper Spacing</span></p>
    <ul>
        <li>Allow adequate airflow between plants</li>
        <li>Avoid overcrowding</li>
    </ul>
    
    <p><span class='highlight'>5. Soil Health</span></p>
    <ul>
        <li>Use quality, well-draining soil</li>
        <li>Add compost regularly</li>
        <li>Test soil pH for specific plant needs</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

