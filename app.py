import streamlit as st
import joblib
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load models
knn = joblib.load("knn_model.pkl")
pca = joblib.load("pca_model.pkl")

# Page configuration
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

# Title and description
st.title("🔢 Handwritten Digit Recognition (MNIST)")
st.write("Predict handwritten digits using our trained KNN model!")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📤 Upload Image", "✏️ Draw Digit"])

# ============================================================================
# TAB 1: Upload Image
# ============================================================================
with tab1:
    st.subheader("Upload a handwritten digit image")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Process uploaded image
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((28, 28))
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", width=200)
        
        with col2:
            # Preprocess and predict
            img = np.array(image).reshape(1, -1) / 255.0
            img_pca = pca.transform(img)
            prediction = knn.predict(img_pca)
            
            st.markdown("### Prediction Result")
            st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
                       unsafe_allow_html=True)
            st.success(f"The model predicts this is digit: **{prediction[0]}**")

# ============================================================================
# TAB 2: Draw Digit
# ============================================================================
with tab2:
    st.subheader("Draw a digit using your mouse")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    with col2:
        st.markdown("### Instructions")
        st.write("1. Draw a digit (0-9) on the canvas")
        st.write("2. Click 'Predict' button below")
        st.write("3. Clear canvas to try again")
        
        # Predict button
        if st.button("🔮 Predict Drawn Digit", type="primary"):
            if canvas_result.image_data is not None:
                # Get the drawn image
                img_data = canvas_result.image_data
                
                # Convert RGBA to grayscale using PIL instead of OpenCV
                # Extract RGB channels (ignore alpha)
                img_rgb = img_data[:, :, :3].astype('uint8')
                
                # Convert to PIL Image
                pil_img = Image.fromarray(img_rgb)
                
                # Convert to grayscale
                img_gray = pil_img.convert('L')
                
                # Resize to 28x28
                img_resized = img_gray.resize((28, 28), Image.LANCZOS)
                
                # Display processed image
                st.image(img_resized, caption="Processed Image (28x28)", width=150)
                
                # Convert to numpy array and preprocess for model
                img_array = np.array(img_resized)
                img_normalized = img_array.reshape(1, -1) / 255.0
                img_pca = pca.transform(img_normalized)
                
                # Predict
                prediction = knn.predict(img_pca)
                
                # Show result
                st.markdown("### Prediction Result")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
                           unsafe_allow_html=True)
                st.success(f"The model predicts: **{prediction[0]}**")
            else:
                st.warning("⚠️ Please draw something on the canvas first!")

# ============================================================================
# Sidebar - Model Information
# ============================================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a **K-Nearest Neighbors (KNN)** classifier 
    trained on the MNIST dataset.
    
    **Model Details:**
    - Algorithm: KNN
    - Dimensionality Reduction: PCA (50 components)
    - Training Accuracy: 97.3%
    
    **Dataset:**
    - 60,000 training images
    - 10,000 test images
    - Image size: 28×28 pixels
    """)
    
    st.header("🎯 Tips for Best Results")
    st.write("""
    - Draw digits clearly and centered
    - Use the full canvas area
    - Make strokes thick and bold
    - Try to match MNIST writing style
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with Streamlit | MNIST Digit Recognition</p>", 
    unsafe_allow_html=True
)
