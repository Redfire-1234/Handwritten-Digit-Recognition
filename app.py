import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

# Load models with caching and error handling
@st.cache_resource
def load_models():
    try:
        # Check if model files exist
        if not os.path.exists("knn_model.pkl"):
            st.error("❌ knn_model.pkl not found! Please upload the model file.")
            st.stop()
        if not os.path.exists("pca_model.pkl"):
            st.error("❌ pca_model.pkl not found! Please upload the model file.")
            st.stop()
        
        # Load models
        knn = joblib.load("knn_model.pkl")
        pca = joblib.load("pca_model.pkl")
        
        return knn, pca
    except EOFError:
        st.error("❌ Model files are corrupted or incomplete. Please regenerate and upload them.")
        st.info("💡 Run the training notebook and save models using `joblib.dump()`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# Try to load models
knn, pca = load_models()

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
        try:
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
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# ============================================================================
# TAB 2: Draw Digit
# ============================================================================
with tab2:
    st.subheader("Draw a digit using your mouse")
    
    # Initialize session state for canvas activation
    if 'canvas_ready' not in st.session_state:
        st.session_state.canvas_ready = False
    
    # Show a button to activate canvas (lazy loading)
    if not st.session_state.canvas_ready:
        st.info("👆 Click the button below to activate the drawing canvas")
        if st.button("🎨 Activate Drawing Canvas", type="primary"):
            st.session_state.canvas_ready = True
            st.rerun()
    
    # Load canvas only when activated
    if st.session_state.canvas_ready:
        try:
            from streamlit_drawable_canvas import st_canvas
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show loading message
                with st.spinner("Loading canvas..."):
                    # Create a canvas component with optimized settings
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 0, 0, 1)",
                        stroke_width=25,
                        stroke_color="#FFFFFF",
                        background_color="#000000",
                        background_image=None,
                        update_streamlit=True,
                        height=280,
                        width=280,
                        drawing_mode="freedraw",
                        point_display_radius=0,
                        display_toolbar=True,
                        key="canvas",
                    )
            
            with col2:
                st.markdown("### Instructions")
                st.write("1. Draw a digit (0-9) on the canvas")
                st.write("2. Click 'Predict' button")
                st.write("3. Use 🗑️ to clear")
                
                # Add a reset button
                if st.button("🔄 Reset Canvas"):
                    st.session_state.canvas_ready = False
                    st.rerun()
                
                st.markdown("---")
                
                # Predict button
                if st.button("🔮 Predict Drawn Digit", type="primary"):
                    if canvas_result.image_data is not None:
                        # Check if something is drawn
                        if np.sum(canvas_result.image_data[:, :, :3]) > 0:
                            try:
                                with st.spinner("Analyzing..."):
                                    # Get the drawn image
                                    img_data = canvas_result.image_data
                                    
                                    # Convert RGBA to grayscale using PIL
                                    img_rgb = img_data[:, :, :3].astype('uint8')
                                    pil_img = Image.fromarray(img_rgb)
                                    img_gray = pil_img.convert('L')
                                    
                                    # Resize to 28x28
                                    img_resized = img_gray.resize((28, 28), Image.LANCZOS)
                                    
                                    # Display processed image
                                    st.image(img_resized, caption="Processed (28x28)", width=150)
                                    
                                    # Convert to numpy array and preprocess for model
                                    img_array = np.array(img_resized)
                                    img_normalized = img_array.reshape(1, -1) / 255.0
                                    img_pca = pca.transform(img_normalized)
                                    
                                    # Predict
                                    prediction = knn.predict(img_pca)
                                    
                                    # Show result
                                    st.markdown("### Result")
                                    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
                                               unsafe_allow_html=True)
                                    st.success(f"Predicted: **{prediction[0]}**")
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
                        else:
                            st.warning("⚠️ Canvas is empty!")
                    else:
                        st.warning("⚠️ Please draw first!")
        
        except ImportError:
            st.error("❌ Canvas feature unavailable. Install: `pip install streamlit-drawable-canvas`")
    
    # Additional option: Upload drawn image (Always visible)
    st.markdown("---")
    st.markdown("### 📎 Or Upload Your Drawing")
    st.caption("Faster alternative - draw on paper/app and upload")
    
    drawn_file = st.file_uploader("Upload your hand-drawn digit", type=["png", "jpg", "jpeg"], key="drawn_upload")
    
    if drawn_file:
        try:
            # Process uploaded drawn image
            image = Image.open(drawn_file).convert("L")
            
            # Invert if needed (if drawing is black on white)
            img_array = np.array(image)
            if np.mean(img_array) > 127:  # More white than black
                image = Image.fromarray(255 - img_array)
            
            image = image.resize((28, 28))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Your Drawing (28x28)", width=200)
            
            with col2:
                # Preprocess and predict
                img = np.array(image).reshape(1, -1) / 255.0
                img_pca = pca.transform(img)
                prediction = knn.predict(img_pca)
                
                st.markdown("### Prediction Result")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
                           unsafe_allow_html=True)
                st.success(f"The model predicts: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

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
    
    st.header("⚡ Performance Tips")
    st.write("""
    **Canvas Loading Slow?**
    - Use "Upload Your Drawing" instead
    - It's much faster!
    
    **For Best Results:**
    - Draw clearly and centered
    - Use thick, bold strokes
    - White digit on black background
    """)
    
    # Show model status
    st.header("🔧 Model Status")
    if os.path.exists("knn_model.pkl") and os.path.exists("pca_model.pkl"):
        st.success("✓ Models loaded successfully")
        knn_size = os.path.getsize("knn_model.pkl") / (1024*1024)
        pca_size = os.path.getsize("pca_model.pkl") / (1024*1024)
        st.caption(f"KNN: {knn_size:.2f} MB")
        st.caption(f"PCA: {pca_size:.2f} MB")
    else:
        st.error("✗ Model files missing")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with Streamlit | MNIST Digit Recognition</p>", 
    unsafe_allow_html=True
)
