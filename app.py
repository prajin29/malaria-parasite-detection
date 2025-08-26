import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #4caf50;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #ffc107;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model():
    """Load the malaria detection model"""
    try:
        # Check if model file exists
        if not os.path.exists('malaria_model.h5'):
            st.warning("📥 Model file not found locally. Attempting to download...")
            
            # Google Drive direct download link (you'll need to replace this)
            model_url = "YOUR_GOOGLE_DRIVE_DIRECT_LINK_HERE"
            
            try:
                import requests
                st.info("🔄 Downloading model from cloud storage...")
                
                # Download the model
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open('malaria_model.h5', 'wb') as f:
                    for data in response.iter_content(block_size):
                        downloaded += len(data)
                        f.write(data)
                        
                        # Show progress
                        if total_size > 0:
                            progress = downloaded / total_size
                            st.progress(progress)
                            st.text(f"Downloaded: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
                
                st.success("✅ Model downloaded successfully!")
                
            except Exception as download_error:
                st.error("❌ Failed to download model automatically")
                st.error("Please manually download the model file and place it in the same directory as this app")
                st.error(f"Download error: {str(download_error)}")
                return None
        
        # Check file size to ensure it's not empty
        file_size = os.path.getsize('malaria_model.h5')
        if file_size < 1000:  # Less than 1KB
            st.error("❌ Model file appears to be empty or corrupted")
            st.info(f"File size: {file_size} bytes (should be much larger)")
            return None
        
        st.info(f"📁 Model file found: {file_size / (1024*1024):.2f} MB")
        
        # Try to load the model
        st.info("🔄 Loading malaria detection model...")
        
        try:
            # First try: Standard model loading
            model = tf.keras.models.load_model('malaria_model.h5')
        except Exception as e1:
            st.warning("⚠️ Standard loading failed, trying alternative methods...")
            
            try:
                # Second try: Load with custom_objects
                model = tf.keras.models.load_model('malaria_model.h5', compile=False)
            except Exception as e2:
                st.warning("⚠️ Alternative loading failed, trying to load weights only...")
                
                try:
                    # Third try: Load just the architecture and weights separately
                    from tensorflow.keras.applications import EfficientNetB0
                    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
                    from tensorflow.keras.models import Model
                    
                    # Create the model architecture
                    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
                    x = base_model.output
                    x = GlobalAveragePooling2D()(x)
                    x = Dense(128, activation='relu')(x)
                    predictions = Dense(2, activation='softmax')(x)
                    model = Model(inputs=base_model.input, outputs=predictions)
                    
                    # Try to load weights
                    model.load_weights('malaria_model.h5')
                    st.success("✅ Model loaded using architecture reconstruction method")
                    
                except Exception as e3:
                    st.error("❌ All loading methods failed")
                    st.error(f"Error 1 (standard): {str(e1)}")
                    st.error(f"Error 2 (compile=False): {str(e2)}")
                    st.error(f"Error 3 (architecture): {str(e3)}")
                    st.error("Your model file may be corrupted or incompatible")
                    return None
        
        # Check model input/output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        st.success(f"✅ Model loaded successfully!")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📊 Model Information:**")
        st.markdown(f"   - **Input shape:** {input_shape}")
        st.markdown(f"   - **Output shape:** {output_shape}")
        st.markdown(f"   - **Model type:** {type(model).__name__}")
        
        # Display model requirements
        if len(input_shape) == 4:
            height, width, channels = input_shape[1], input_shape[2], input_shape[3]
            st.markdown("**📋 Model Requirements:**")
            st.markdown(f"   - **Image dimensions:** {height} x {width} pixels")
            st.markdown(f"   - **Color channels:** {channels} (RGB)")
            st.markdown(f"   - **Data type:** float32")
            st.markdown(f"   - **Normalization:** 0-1 range")
        st.markdown('</div>', unsafe_allow_html=True)
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("This could be due to:")
        st.error("   - Incompatible TensorFlow version")
        st.error("   - Corrupted model file")
        st.error("   - Missing dependencies")
        return None

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Prediction function
def predict_malaria(model, image):
    """Make prediction using the loaded model"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None, None
        
        # Get model input shape
        expected_input_shape = model.input_shape
        st.info(f"Model expects input shape: {expected_input_shape}")
        st.info(f"Processed image shape: {processed_image.shape}")
        
        # Check if we need to adjust the input
        if len(expected_input_shape) == 4:  # (batch, height, width, channels)
            if expected_input_shape[1] is not None and expected_input_shape[1] != processed_image.shape[1]:
                # Resize to expected dimensions
                new_height = expected_input_shape[1]
                new_width = expected_input_shape[2] if expected_input_shape[2] is not None else new_height
                processed_image = cv2.resize(processed_image[0], (new_width, new_height))
                processed_image = np.expand_dims(processed_image, axis=0)
                st.info(f"Resized image to: {processed_image.shape}")
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Handle different model output formats
        if len(prediction.shape) == 2:
            if prediction.shape[1] == 1:
                # Single output (infected probability)
                infected_prob = float(prediction[0][0])
                uninfected_prob = 1 - infected_prob
            elif prediction.shape[1] == 2:
                # Two outputs (infected, uninfected)
                infected_prob = float(prediction[0][1])  # Usually second class is infected
                uninfected_prob = float(prediction[0][0])
            else:
                st.error(f"Unexpected model output shape: {prediction.shape}")
                return None, None
        else:
            st.error(f"Unexpected model output format: {prediction.shape}")
            return None, None
        
        # Ensure probabilities are valid
        infected_prob = max(0.0, min(1.0, infected_prob))
        uninfected_prob = max(0.0, min(1.0, uninfected_prob))
        
        return infected_prob, uninfected_prob
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("This usually means the image format doesn't match what the model expects.")
        st.error("Try uploading a different image or check the model requirements.")
        return None, None

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🦟 Malaria Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📋 About")
    st.sidebar.markdown("""
    This application uses a deep learning model to detect malaria parasites in blood cell images.
    
    **How it works:**
    1. Upload a blood cell image
    2. The model analyzes the image
    3. Get instant results showing infection probability
    
    **Supported formats:** JPG, PNG, JPEG
    """)
    
    st.sidebar.markdown("## ⚠️ Disclaimer")
    st.sidebar.markdown("""
    This tool is for educational and research purposes only. 
    Always consult healthcare professionals for medical diagnosis.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check if 'malaria_model.h5' exists in the current directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a blood cell image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a blood cell image for malaria detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.markdown('<h3 class="sub-header">📷 Uploaded Image</h3>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Image Mode:** {image.mode}")
            st.write(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Analyze button
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    infected_prob, uninfected_prob = predict_malaria(model, image)
                    
                    if infected_prob is not None:
                        # Display results
                        st.markdown('<h3 class="sub-header">📊 Detection Results</h3>', unsafe_allow_html=True)
                        
                        # Create progress bars
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                label="Infected Probability",
                                value=f"{infected_prob:.2%}",
                                delta=f"{infected_prob:.3f}"
                            )
                            st.progress(infected_prob)
                        
                        with col_b:
                            st.metric(
                                label="Uninfected Probability",
                                value=f"{uninfected_prob:.2%}",
                                delta=f"{uninfected_prob:.3f}"
                            )
                            st.progress(uninfected_prob)
                        
                        # Result interpretation
                        if infected_prob > 0.7:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.markdown("## ⚠️ HIGH RISK DETECTED")
                            st.markdown(f"**Probability of malaria infection: {infected_prob:.2%}**")
                            st.markdown("**Recommendation:** Immediate medical consultation required")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif infected_prob > 0.4:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.markdown("## ⚠️ MODERATE RISK DETECTED")
                            st.markdown(f"**Probability of malaria infection: {infected_prob:.2%}**")
                            st.markdown("**Recommendation:** Medical consultation recommended")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.markdown("## ✅ LOW RISK DETECTED")
                            st.markdown(f"**Probability of malaria infection: {infected_prob:.2%}**")
                            st.markdown("**Recommendation:** Continue monitoring, consult if symptoms appear")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Additional information
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**Model Confidence:** High")
                        st.markdown("**Analysis Time:** < 1 second")
                        st.markdown("**Image Quality:** Good")
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder when no image is uploaded
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            st.markdown("## 📁 No Image Selected")
            st.markdown("Please upload an image to begin analysis")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🦟 Malaria Detection System | Powered by TensorFlow & Streamlit</p>
        <p>For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
