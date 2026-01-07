import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
from PIL import Image
import os
import io
import base64

# Page configuration
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

# Load models with caching and error handling
@st.cache_resource
def load_models():
    try:
        if not os.path.exists("knn_model.pkl"):
            st.error("❌ knn_model.pkl not found!")
            st.stop()
        if not os.path.exists("pca_model.pkl"):
            st.error("❌ pca_model.pkl not found!")
            st.stop()
        
        knn = joblib.load("knn_model.pkl")
        pca = joblib.load("pca_model.pkl")
        return knn, pca
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

knn, pca = load_models()

# Title
st.title("🔢 Handwritten Digit Recognition (MNIST)")
st.write("Predict handwritten digits using our trained KNN model!")

# Create tabs
tab1, tab2 = st.tabs(["📤 Upload Image", "✏️ Draw Digit"])

# ============================================================================
# TAB 1: Upload Image
# ============================================================================
with tab1:
    st.subheader("Upload a handwritten digit image")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("L")
            image = image.resize((28, 28))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", width=200)
            
            with col2:
                img = np.array(image).reshape(1, -1) / 255.0
                img_pca = pca.transform(img)
                prediction = knn.predict(img_pca)
                
                st.markdown("### Prediction Result")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
                           unsafe_allow_html=True)
                st.success(f"The model predicts: **{prediction[0]}**")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================================================
# TAB 2: Draw Digit (Fast HTML5 Canvas)
# ============================================================================
with tab2:
    st.subheader("Draw a digit using your mouse")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fast HTML5 Canvas with JavaScript
        canvas_html = """
        <style>
        #canvas {
            border: 2px solid #4CAF50;
            cursor: crosshair;
            background-color: black;
            border-radius: 8px;
        }
        .canvas-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }
        .btn-group {
            display: flex;
            gap: 10px;
        }
        button {
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        #clearBtn {
            background-color: #f44336;
            color: white;
        }
        #clearBtn:hover {
            background-color: #da190b;
        }
        #saveBtn {
            background-color: #4CAF50;
            color: white;
        }
        #saveBtn:hover {
            background-color: #45a049;
        }
        </style>

        <div class="canvas-container">
            <canvas id="canvas" width="280" height="280"></canvas>
            <div class="btn-group">
                <button id="clearBtn">🗑️ Clear Canvas</button>
                <button id="saveBtn">💾 Save Drawing</button>
            </div>
            <div id="status" style="margin-top: 10px; font-weight: bold;"></div>
        </div>

        <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const clearBtn = document.getElementById('clearBtn');
        const saveBtn = document.getElementById('saveBtn');
        const status = document.getElementById('status');
        
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        // Initialize canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        function startDrawing(e) {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            lastX = e.clientX - rect.left;
            lastY = e.clientY - rect.top;
        }

        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(currentX, currentY);
            ctx.stroke();
            
            lastX = currentX;
            lastY = currentY;
        }

        function stopDrawing() {
            isDrawing = false;
        }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            status.textContent = 'Canvas cleared!';
            status.style.color = '#666';
            setTimeout(() => status.textContent = '', 2000);
        }

        function saveDrawing() {
            const imageData = canvas.toDataURL('image/png');
            // Send to Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                data: imageData
            }, '*');
            status.textContent = '✓ Drawing saved! Click Predict below.';
            status.style.color = '#4CAF50';
        }

        // Mouse events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });

        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });

        canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            canvas.dispatchEvent(mouseEvent);
        });

        // Button events
        clearBtn.addEventListener('click', clearCanvas);
        saveBtn.addEventListener('click', saveDrawing);
        </script>
        """
        
        # Render the canvas (loads instantly!)
        canvas_result = components.html(canvas_html, height=400)
    
    with col2:
        st.markdown("### Instructions")
        st.write("1. **Draw** a digit (0-9) on canvas")
        st.write("2. Click **💾 Save Drawing**")
        st.write("3. Click **🔮 Predict** below")
        st.write("4. Use **🗑️ Clear** to restart")
        
        st.markdown("---")
        
        # Get the canvas data from component
        if canvas_result:
            try:
                # Decode base64 image
                image_data = canvas_result.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to grayscale
                image = image.convert('L')
                
                # Resize to 28x28
                image_resized = image.resize((28, 28), Image.LANCZOS)
                
                # Show processed image
                st.image(image_resized, caption="Processed (28x28)", width=150)
                
                # Predict
                img_array = np.array(image_resized)
                img_normalized = img_array.reshape(1, -1) / 255.0
                img_pca = pca.transform(img_normalized)
                prediction = knn.predict(img_pca)
                
                # Show result
                st.markdown("### Result")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
                           unsafe_allow_html=True)
                st.success(f"Predicted: **{prediction[0]}**")
                
            except Exception as e:
                st.info("Draw and save your digit to see prediction")
        
        # Alternative: Manual predict button with file uploader
        st.markdown("---")
        st.markdown("### 📎 Or Upload Drawing")
        drawn_file = st.file_uploader("Upload hand-drawn digit", type=["png", "jpg", "jpeg"], key="drawn")
        
        if drawn_file:
            try:
                image = Image.open(drawn_file).convert("L")
                
                # Invert if needed
                img_array = np.array(image)
                if np.mean(img_array) > 127:
                    image = Image.fromarray(255 - img_array)
                
                image = image.resize((28, 28))
                
                st.image(image, caption="Uploaded (28x28)", width=150)
                
                # Predict
                img = np.array(image).reshape(1, -1) / 255.0
                img_pca = pca.transform(img)
                prediction = knn.predict(img_pca)
                
                st.markdown("### Result")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
                           unsafe_allow_html=True)
                st.success(f"Predicted: **{prediction[0]}**")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **K-Nearest Neighbors (KNN)** classifier trained on MNIST dataset.
    
    **Model Details:**
    - Algorithm: KNN
    - PCA: 50 components
    - Accuracy: 97.3%
    
    **Dataset:**
    - 60,000 training images
    - 10,000 test images
    - Size: 28×28 pixels
    """)
    
    st.header("🎯 Tips")
    st.write("""
    - Draw clearly and centered
    - Use thick, bold strokes
    - White on black background
    - Click Save before Predict
    """)
    
    # Model status
    st.header("🔧 Status")
    if os.path.exists("knn_model.pkl") and os.path.exists("pca_model.pkl"):
        st.success("✓ Models loaded")
    else:
        st.error("✗ Models missing")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with Streamlit</p>", unsafe_allow_html=True)
# import streamlit as st
# import joblib
# import numpy as np
# from PIL import Image
# import os

# # Page configuration
# st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

# # Load models with caching and error handling
# @st.cache_resource
# def load_models():
#     try:
#         # Check if model files exist
#         if not os.path.exists("knn_model.pkl"):
#             st.error("❌ knn_model.pkl not found! Please upload the model file.")
#             st.stop()
#         if not os.path.exists("pca_model.pkl"):
#             st.error("❌ pca_model.pkl not found! Please upload the model file.")
#             st.stop()
        
#         # Load models
#         knn = joblib.load("knn_model.pkl")
#         pca = joblib.load("pca_model.pkl")
        
#         return knn, pca
#     except EOFError:
#         st.error("❌ Model files are corrupted or incomplete. Please regenerate and upload them.")
#         st.info("💡 Run the training notebook and save models using `joblib.dump()`")
#         st.stop()
#     except Exception as e:
#         st.error(f"❌ Error loading models: {str(e)}")
#         st.stop()

# # Try to load models
# knn, pca = load_models()

# # Title and description
# st.title("🔢 Handwritten Digit Recognition (MNIST)")
# st.write("Predict handwritten digits using our trained KNN model!")

# # Create tabs for different input methods
# tab1, tab2 = st.tabs(["📤 Upload Image", "✏️ Draw Digit"])

# # ============================================================================
# # TAB 1: Upload Image
# # ============================================================================
# with tab1:
#     st.subheader("Upload a handwritten digit image")
#     uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
#     if uploaded_file:
#         try:
#             # Process uploaded image
#             image = Image.open(uploaded_file).convert("L")
#             image = image.resize((28, 28))
            
#             # Display in columns
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.image(image, caption="Uploaded Image", width=200)
            
#             with col2:
#                 # Preprocess and predict
#                 img = np.array(image).reshape(1, -1) / 255.0
#                 img_pca = pca.transform(img)
#                 prediction = knn.predict(img_pca)
                
#                 st.markdown("### Prediction Result")
#                 st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
#                            unsafe_allow_html=True)
#                 st.success(f"The model predicts this is digit: **{prediction[0]}**")
#         except Exception as e:
#             st.error(f"Error processing image: {str(e)}")

# # ============================================================================
# # TAB 2: Draw Digit
# # ============================================================================
# with tab2:
#     st.subheader("Draw a digit using your mouse")
    
#     # Initialize session state for canvas activation
#     if 'canvas_ready' not in st.session_state:
#         st.session_state.canvas_ready = False
    
#     # Show a button to activate canvas (lazy loading)
#     if not st.session_state.canvas_ready:
#         st.info("👆 Click the button below to activate the drawing canvas")
#         if st.button("🎨 Activate Drawing Canvas", type="primary"):
#             st.session_state.canvas_ready = True
#             st.rerun()
    
#     # Load canvas only when activated
#     if st.session_state.canvas_ready:
#         try:
#             from streamlit_drawable_canvas import st_canvas
            
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 # Show loading message
#                 with st.spinner("Loading canvas..."):
#                     # Create a canvas component with optimized settings
#                     canvas_result = st_canvas(
#                         fill_color="rgba(0, 0, 0, 1)",
#                         stroke_width=25,
#                         stroke_color="#FFFFFF",
#                         background_color="#000000",
#                         background_image=None,
#                         update_streamlit=True,
#                         height=280,
#                         width=280,
#                         drawing_mode="freedraw",
#                         point_display_radius=0,
#                         display_toolbar=True,
#                         key="canvas",
#                     )
            
#             with col2:
#                 st.markdown("### Instructions")
#                 st.write("1. Draw a digit (0-9) on the canvas")
#                 st.write("2. Click 'Predict' button")
#                 st.write("3. Use 🗑️ to clear")
                
#                 # Add a reset button
#                 if st.button("🔄 Reset Canvas"):
#                     st.session_state.canvas_ready = False
#                     st.rerun()
                
#                 st.markdown("---")
                
#                 # Predict button
#                 if st.button("🔮 Predict Drawn Digit", type="primary"):
#                     if canvas_result.image_data is not None:
#                         # Check if something is drawn
#                         if np.sum(canvas_result.image_data[:, :, :3]) > 0:
#                             try:
#                                 with st.spinner("Analyzing..."):
#                                     # Get the drawn image
#                                     img_data = canvas_result.image_data
                                    
#                                     # Convert RGBA to grayscale using PIL
#                                     img_rgb = img_data[:, :, :3].astype('uint8')
#                                     pil_img = Image.fromarray(img_rgb)
#                                     img_gray = pil_img.convert('L')
                                    
#                                     # Resize to 28x28
#                                     img_resized = img_gray.resize((28, 28), Image.LANCZOS)
                                    
#                                     # Display processed image
#                                     st.image(img_resized, caption="Processed (28x28)", width=150)
                                    
#                                     # Convert to numpy array and preprocess for model
#                                     img_array = np.array(img_resized)
#                                     img_normalized = img_array.reshape(1, -1) / 255.0
#                                     img_pca = pca.transform(img_normalized)
                                    
#                                     # Predict
#                                     prediction = knn.predict(img_pca)
                                    
#                                     # Show result
#                                     st.markdown("### Result")
#                                     st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
#                                                unsafe_allow_html=True)
#                                     st.success(f"Predicted: **{prediction[0]}**")
#                             except Exception as e:
#                                 st.error(f"Error during prediction: {str(e)}")
#                         else:
#                             st.warning("⚠️ Canvas is empty!")
#                     else:
#                         st.warning("⚠️ Please draw first!")
        
#         except ImportError:
#             st.error("❌ Canvas feature unavailable. Install: `pip install streamlit-drawable-canvas`")
    
#     # Additional option: Upload drawn image (Always visible)
#     st.markdown("---")
#     st.markdown("### 📎 Or Upload Your Drawing")
#     st.caption("Faster alternative - draw on paper/app and upload")
    
#     drawn_file = st.file_uploader("Upload your hand-drawn digit", type=["png", "jpg", "jpeg"], key="drawn_upload")
    
#     if drawn_file:
#         try:
#             # Process uploaded drawn image
#             image = Image.open(drawn_file).convert("L")
            
#             # Invert if needed (if drawing is black on white)
#             img_array = np.array(image)
#             if np.mean(img_array) > 127:  # More white than black
#                 image = Image.fromarray(255 - img_array)
            
#             image = image.resize((28, 28))
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.image(image, caption="Your Drawing (28x28)", width=200)
            
#             with col2:
#                 # Preprocess and predict
#                 img = np.array(image).reshape(1, -1) / 255.0
#                 img_pca = pca.transform(img)
#                 prediction = knn.predict(img_pca)
                
#                 st.markdown("### Prediction Result")
#                 st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{prediction[0]}</h1>", 
#                            unsafe_allow_html=True)
#                 st.success(f"The model predicts: **{prediction[0]}**")
#         except Exception as e:
#             st.error(f"Error processing image: {str(e)}")

# # ============================================================================
# # Sidebar - Model Information
# # ============================================================================
# with st.sidebar:
#     st.header("ℹ️ About")
#     st.write("""
#     This application uses a **K-Nearest Neighbors (KNN)** classifier 
#     trained on the MNIST dataset.
    
#     **Model Details:**
#     - Algorithm: KNN
#     - Dimensionality Reduction: PCA (50 components)
#     - Training Accuracy: 97.3%
    
#     **Dataset:**
#     - 60,000 training images
#     - 10,000 test images
#     - Image size: 28×28 pixels
#     """)
    
#     st.header("⚡ Performance Tips")
#     st.write("""
#     **Canvas Loading Slow?**
#     - Use "Upload Your Drawing" instead
#     - It's much faster!
    
#     **For Best Results:**
#     - Draw clearly and centered
#     - Use thick, bold strokes
#     - White digit on black background
#     """)
    
#     # Show model status
#     st.header("🔧 Model Status")
#     if os.path.exists("knn_model.pkl") and os.path.exists("pca_model.pkl"):
#         st.success("✓ Models loaded successfully")
#         knn_size = os.path.getsize("knn_model.pkl") / (1024*1024)
#         pca_size = os.path.getsize("pca_model.pkl") / (1024*1024)
#         st.caption(f"KNN: {knn_size:.2f} MB")
#         st.caption(f"PCA: {pca_size:.2f} MB")
#     else:
#         st.error("✗ Model files missing")

# # Footer
# st.markdown("---")
# st.markdown(
#     "<p style='text-align: center; color: gray;'>Built with Streamlit | MNIST Digit Recognition</p>", 
#     unsafe_allow_html=True
# )
