

import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np
import torch
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Sri Vijay Vidhyalaya ANPR",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f6f8fb 0%, #e9f1f7 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card {
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .feature-box {
        padding: 1rem;
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .highlight-text {
        color: #1f77b4;
        font-weight: bold;
    }
    .exit-button {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .model-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar college logo
try:
    logo = Image.open('college_logo.png')
    st.sidebar.image(logo, width=150)
except:
    st.sidebar.image("C:/Users/Hirthick/Dharmapuri Projects/Automatic/Streamlit/img.png", width=300)

# Sidebar navigation - MUST BE DEFINED BEFORE USE
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
# Define the navigation variable here
selected_page = st.sidebar.radio(
    "Go to",
    ["Home", "Model Implementation", "Exit"],
    key="navigation"
)

# Sidebar About section
st.sidebar.markdown("### About")
st.sidebar.info("""
    **Developer:** G.Palaniyammal
                    
    **College:** Sri Vijay Vidyalaya College of Arts and Science  

    This application is built using Streamlit 
""")

# Model loading function
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="model/last.pt", force_reload=True)
    model.eval()
    model.conf = 0.5
    model.iou = 0.45
    return model

def home_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Automatic Number Plate Detection System")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h2>Automatic Number Plate Detection System(ANPD)</h2>
        <p>Automatic Number Plate Recognition (ANPD) is a cutting-edge technology that uses optical character recognition 
        (OCR) and computer vision techniques to automatically read vehicle registration plates. Our system employs 
        advanced deep learning models to deliver accurate and reliable results in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>Key Features</h3>
            <div class="feature-box">✨ Real-time plate detection</div>
            <div class="feature-box">🎯 High accuracy recognition</div>
            <div class="feature-box">🚀 Fast processing speed</div>
            <div class="feature-box">📱 Multi-platform support</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>Applications</h3>
            <div class="feature-box">🏢 Smart parking systems</div>
            <div class="feature-box">🛣️ Traffic management</div>
            <div class="feature-box">🔒 Security and surveillance</div>
            <div class="feature-box">🏭 Industrial automation</div>
        </div>
        """, unsafe_allow_html=True)

def model_implementation_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Model Implementation")
    st.markdown('</div>', unsafe_allow_html=True)

    tabs = st.tabs(["📷 Image Upload", "🎥 Live Camera"])
    
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload vehicle image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            try:
                # Create directories if they don't exist
                os.makedirs("uploads", exist_ok=True)
                os.makedirs("downloads", exist_ok=True)
                
                # Save and process image
                image_path = os.path.join("uploads", uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and process image
                image = Image.open(uploaded_file)
                model = load_model()
                results = model(image, size=640)
                
                # Display results
                st.image(results.render()[0], caption="Processed Image", use_column_width=True)
                
                # Download button
                processed_image = Image.fromarray(results.render()[0])
                buffered = BytesIO()
                processed_image.save(buffered, format="JPEG")
                
                st.download_button(
                    label="Download Processed Image",
                    data=buffered.getvalue(),
                    file_name=f"processed_{uploaded_file.name}",
                    mime=f"image/{uploaded_file.name.split('.')[-1].lower()}"
                )
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.warning("⚠️ Please ensure camera access is enabled")
        start_camera = st.checkbox("Start Camera")
        
        if start_camera:
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access camera")
                        break
                    
                    # Process frame
                    model = load_model()
                    results = model(frame)
                    processed_frame = results.render()[0]
                    
                    # Display frame
                    frame_placeholder.image(processed_frame, channels="BGR")
                    
                    if not start_camera:
                        break
                        
            finally:
                cap.release()

def exit_page():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("👋 Thank You for Using Our ANPD System!")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>Contact Information</h4>
            <p>Department of Computer Science<br>
            Sri Vijay Vidhyalaya College of Arts and Science<br>
            Dharmapuri<br>
            Tamil Nadu, India</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>Developer</h4>
            <p>G.Palaniyammal<br>
            Computer Science Department<br>
            Batch of 2024</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Exit Application", key="exit_button"):
        st.balloons()
        st.success("Thank you for using our ANPD system! You can close this window.")
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem;'>
            <h3>🎉 Visit Again! 🎉</h3>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

# Page routing using the selected_page variable
if selected_page == "Home":
    home_page()
elif selected_page == "Model Implementation":
    model_implementation_page()
else:
    exit_page()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p>© 2024 Sri Vijay Vidhyalaya College of Arts and Science, Dharmapuri</p>
        <p style='color: #666; font-size: 0.8rem;'>Powered by Streamlit & PyTorch</p>
    </div>
""", unsafe_allow_html=True)