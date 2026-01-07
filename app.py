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
# TAB 2: Draw Digit (Compact Layout)
# ============================================================================
with tab2:
    # Info expander (collapsible)
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        **Steps:**
        1. Draw a digit (0-9) on the canvas
        2. Click **💾 Download Image** button
        3. Upload the downloaded image in the section below
        4. Click **🔮 Predict** to see the result
        
        **Tips:** Draw clearly with thick strokes, centered on canvas
        """)
    
    # Two columns: Canvas + Upload/Predict
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("##### 🎨 Drawing Canvas")
        # Fast HTML5 Canvas
        canvas_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body { margin: 0; padding: 10px; font-family: sans-serif; }
        #canvas {
            border: 3px solid #4CAF50;
            cursor: crosshair;
            background-color: black;
            border-radius: 8px;
            display: block;
        }
        .btn-group {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        button {
            padding: 10px 18px;
            font-size: 14px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            flex: 1;
        }
        #clearBtn { background-color: #f44336; color: white; }
        #clearBtn:hover { background-color: #da190b; }
        #downloadBtn { background-color: #4CAF50; color: white; }
        #downloadBtn:hover { background-color: #45a049; }
        #status {
            margin-top: 8px;
            font-size: 13px;
            font-weight: bold;
            min-height: 18px;
            text-align: center;
        }
        </style>
        </head>
        <body>
            <canvas id="canvas" width="300" height="300"></canvas>
            <div class="btn-group">
                <button id="clearBtn">🗑️ Clear</button>
                <button id="downloadBtn">💾 Download</button>
            </div>
            <div id="status"></div>

        <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const clearBtn = document.getElementById('clearBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const status = document.getElementById('status');
        
        let isDrawing = false;
        let lastX = 0, lastY = 0;

        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        function startDrawing(e) {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            lastX = (e.clientX - rect.left) * scaleX;
            lastY = (e.clientY - rect.top) * scaleY;
        }

        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const currentX = (e.clientX - rect.left) * scaleX;
            const currentY = (e.clientY - rect.top) * scaleY;
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(currentX, currentY);
            ctx.stroke();
            
            lastX = currentX;
            lastY = currentY;
        }

        function stopDrawing() { isDrawing = false; }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            status.textContent = '✓ Cleared!';
            status.style.color = '#666';
            setTimeout(() => status.textContent = '', 1500);
        }

        function downloadImage() {
            const link = document.createElement('a');
            link.download = 'digit_drawing.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            status.textContent = '✓ Downloaded! Upload below ↓';
            status.style.color = '#4CAF50';
        }

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            startDrawing({clientX: touch.clientX, clientY: touch.clientY});
        });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            draw({clientX: touch.clientX, clientY: touch.clientY});
        });
        canvas.addEventListener('touchend', stopDrawing);

        clearBtn.addEventListener('click', clearCanvas);
        downloadBtn.addEventListener('click', downloadImage);
        </script>
        </body>
        </html>
        """
        
        components.html(canvas_html, height=410, scrolling=False)
    
    with col2:
        st.markdown("##### 📤 Upload & Predict")
        
        # Tabbed interface for upload options
        upload_tab1, upload_tab2 = st.tabs(["Canvas Drawing", "Other Image"])
        
        with upload_tab1:
            drawn_file = st.file_uploader("Upload downloaded image", type=["png", "jpg", "jpeg"], key="canvas_upload", label_visibility="collapsed")
            
            if drawn_file:
                try:
                    image = Image.open(drawn_file).convert("L")
                    image_resized = image.resize((28, 28), Image.LANCZOS)
                    
                    st.image(image_resized, caption="28x28", width=120)
                    
                    if st.button("🔮 Predict", type="primary", use_container_width=True, key="predict1"):
                        with st.spinner("Analyzing..."):
                            img_array = np.array(image_resized)
                            img_normalized = img_array.reshape(1, -1) / 255.0
                            img_pca = pca.transform(img_normalized)
                            prediction = knn.predict(img_pca)
                            
                            st.markdown(f"<h1 style='text-align: center; color: #4CAF50; margin: 10px 0;'>{prediction[0]}</h1>", 
                                       unsafe_allow_html=True)
                            st.success(f"Predicted: **{prediction[0]}**")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with upload_tab2:
            alt_file = st.file_uploader("Upload any digit image", type=["png", "jpg", "jpeg"], key="alt_upload", label_visibility="collapsed")
            
            if alt_file:
                try:
                    image = Image.open(alt_file).convert("L")
                    
                    # Auto-invert if needed
                    img_array = np.array(image)
                    if np.mean(img_array) > 127:
                        image = Image.fromarray(255 - img_array)
                    
                    image = image.resize((28, 28))
                    st.image(image, caption="28x28", width=120)
                    
                    if st.button("🔮 Predict", type="primary", use_container_width=True, key="predict2"):
                        img = np.array(image).reshape(1, -1) / 255.0
                        img_pca = pca.transform(img)
                        prediction = knn.predict(img_pca)
                        
                        st.markdown(f"<h1 style='text-align: center; color: #4CAF50; margin: 10px 0;'>{prediction[0]}</h1>", 
                                   unsafe_allow_html=True)
                        st.success(f"Predicted: **{prediction[0]}**")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("""
    **KNN Classifier**
    - PCA: 50 components
    - Accuracy: 97.3%
    - Dataset: MNIST (28×28)
    """)
    
    # Model status
    st.header("🔧 Status")
    if os.path.exists("knn_model.pkl") and os.path.exists("pca_model.pkl"):
        st.success("✓ Models loaded")
    else:
        st.error("✗ Models missing")
    
    st.markdown("---")
    st.caption("Built with Streamlit")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>MNIST Digit Recognition</p>", unsafe_allow_html=True)
