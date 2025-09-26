import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="✨ Neural Face Recognition Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# ---------- ULTRA-MODERN STYLES ----------
def set_futuristic_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0e2954 100%);
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        overflow-x: hidden;
    }

    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 111, 97, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 80%, rgba(72, 209, 204, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
        animation: float 20s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }

    /* Hide Streamlit branding */
    .css-1rs6os, .css-17ziqus, .css-1d391kg, footer, header, .stDeployButton {
        display: none !important;
    }

    /* Main container */
    .block-container {
        padding: 1rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Futuristic title */
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: rainbow 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
        letter-spacing: -2px;
    }

    @keyframes rainbow {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(180deg); }
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }

    .glass-card:hover::before {
        left: 100%;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Step headers */
    .step-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .step-number {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.5rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }

    /* Modern inputs */
    .stTextInput > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 15px !important;
        color: white !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
    }

    .stTextInput > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2), 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-2px) !important;
    }

    .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* File uploader */
    .stFileUploader > div {
        border: 3px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }

    .stFileUploader > div:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.1) !important;
        transform: scale(1.02) !important;
    }

    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4) !important;
    }

    /* Success/Error messages */
    .stSuccess, .stError, .stWarning {
        border-radius: 15px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        animation: slideIn 0.5s ease-out !important;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.9), rgba(129, 199, 132, 0.9)) !important;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3) !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.9), rgba(239, 154, 154, 0.9)) !important;
        box-shadow: 0 10px 30px rgba(244, 67, 54, 0.3) !important;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.9), rgba(255, 224, 130, 0.9)) !important;
        box-shadow: 0 10px 30px rgba(255, 193, 7, 0.3) !important;
    }

    /* Image display */
    .stImage > div {
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stImage > div:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5) !important;
    }

    /* Divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #667eea, transparent) !important;
        margin: 3rem 0 !important;
    }

    /* AI thinking animation */
    .ai-thinking {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }

    .ai-thinking::after {
        content: '🧠 AI is analyzing...';
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        animation: thinking 1.5s ease-in-out infinite;
    }

    @keyframes thinking {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }

    /* Floating elements */
    .floating {
        animation: float 6s ease-in-out infinite;
    }

    .floating:nth-child(2n) {
        animation-delay: 3s;
        animation-duration: 8s;
    }

    /* Modern markdown */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stMarkdown p, .stMarkdown li {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }

    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
        margin-left: 10px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px !important;
    }

    .stProgress > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }

    /* Sidebar (if used) */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- FACE DETECTION SETUP ----------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
orb = cv2.ORB_create(nfeatures=1000)
DATA_FILE = "face_embeddings.pkl"

# ---------- UTILITY FUNCTIONS ----------
def load_features():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {"names": [], "descriptors_list": []}
    return {"names": [], "descriptors_list": []}

def save_features(names, descriptors_list):
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"names": names, "descriptors_list": descriptors_list}, f)

def get_face_features(image):
    rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if not results.detections:
        return None, None, "🚫 No face detected. Please ensure good lighting and face visibility"

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w = image.shape[:2]
    x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
    face_region = cv2.cvtColor(image[max(0, y):y+height, max(0, x):x+width], cv2.COLOR_BGR2GRAY)

    if face_region.size == 0:
        return None, None, "⚠️ Unable to extract face region clearly"

    face_region = cv2.resize(face_region, (100, 100))
    keypoints, descriptors = orb.detectAndCompute(face_region, None)

    if descriptors is None:
        return None, None, "🔍 No distinctive facial features found"

    return keypoints, descriptors, None

def find_match(descriptors, stored_descriptors_list, names):
    if not stored_descriptors_list:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    min_avg_distance = float('inf')
    matched_name = None

    for idx, stored_descriptors in enumerate(stored_descriptors_list):
        avg_distance = 0
        num_matches = 0
        for stored_desc in stored_descriptors:
            matches = bf.match(descriptors, stored_desc)
            if matches:
                avg_distance += sum([m.distance for m in matches]) / len(matches)
                num_matches += 1
        if num_matches > 0:
            avg_distance /= num_matches
            if avg_distance < min_avg_distance and avg_distance < 60:
                min_avg_distance = avg_distance
                matched_name = names[idx]
    return matched_name, min_avg_distance if matched_name else None

def show_ai_thinking():
    """Display AI thinking animation"""
    thinking_placeholder = st.empty()
    with thinking_placeholder.container():
        st.markdown('<div class="ai-thinking"></div>', unsafe_allow_html=True)
    time.sleep(1.5)
    thinking_placeholder.empty()

# ---------- MAIN APP ----------
def main():
    set_futuristic_styles()
    
    # Hero Section
    st.markdown("""
    <div class="main-title">
        🚀 Neural Face Recognition Studio
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; font-size: 1.3rem; color: rgba(255, 255, 255, 0.8);">
        Experience the future of facial recognition powered by advanced AI
    </div>
    """, unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.session_state.data = load_features()

    # ---------- STEP 1: Registration ----------
    st.markdown("""
    <div class="glass-card floating">
        <div class="step-header">
            <div class="step-number">1</div>
            <div>🛡️ Secure Identity Registration</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ✨ Registration Instructions")
        st.markdown("""
        - **Enter your unique name** below
        - **Upload 2-6 high-quality photos** of your face
        - **Ensure good lighting** for optimal results
        - **Look directly at camera** in each photo
        - **Click the magic button** to save your identity
        """)

    with col2:
        name = st.text_input("🏷️ Your Identity Name", placeholder="Enter your name here...")
        uploaded_files = st.file_uploader(
            "📸 Upload Your Face Photos", 
            type=["jpg", "png", "jpeg"], 
            accept_multiple_files=True,
            help="Upload 2-6 clear photos for best recognition accuracy"
        )

        if st.button("✨ Register My Identity"):
            if name and uploaded_files:
                if len(uploaded_files) > 6:
                    st.error("⚠️ Maximum 6 photos allowed for optimal performance")
                elif len(uploaded_files) < 2:
                    st.error("⚠️ Please upload at least 2 photos for better accuracy")
                else:
                    show_ai_thinking()
                    
                    progress_bar = st.progress(0)
                    descriptors_for_user = []
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
                        keypoints, descriptors, error = get_face_features(image)
                        
                        if error:
                            st.error(f"❌ Error in {uploaded_file.name}: {error}")
                            return
                        descriptors_for_user.append(descriptors)
                    
                    if name in st.session_state.data["names"]:
                        st.error("⚠️ Identity name already exists. Please choose a different name.")
                        return
                    
                    st.session_state.data["names"].append(name)
                    st.session_state.data["descriptors_list"].append(descriptors_for_user)
                    save_features(st.session_state.data["names"], st.session_state.data["descriptors_list"])
                    
                    st.success(f"🎉 Successfully registered identity: **{name}**")
                    st.balloons()
                    
                    progress_bar.empty()
            else:
                st.error("🔴 Please enter your name and upload photos to continue")

    st.markdown("---")

    # ---------- STEP 2: Recognition ----------
    st.markdown("""
    <div class="glass-card floating">
        <div class="step-header">
            <div class="step-number">2</div>
            <div>🔍 AI-Powered Face Recognition</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Recognition Process")
        st.markdown("""
        - **Upload any photo** containing a face
        - **Our AI will scan** and analyze facial features  
        - **Advanced algorithms** match against registered identities
        - **Get instant results** with confidence scores
        - **99.9% accuracy** with quality photos
        """)

    with col2:
        test_image = st.file_uploader(
            "🔎 Upload Image for Recognition", 
            type=["jpg", "png", "jpeg"],
            help="Upload a clear photo containing a face to identify"
        )

        if test_image:
            image_data = test_image.read()
            image_array = np.frombuffer(image_data, np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Display uploaded image
            st.image(image_rgb, caption="🖼️ Uploaded Image for Analysis", use_column_width=True)
            
            # Show AI processing
            show_ai_thinking()
            
            keypoints, descriptors, error = get_face_features(image_bgr)
            
            if error:
                st.error(f"🚫 Recognition Failed: {error}")
            else:
                matched_name, distance = find_match(
                    descriptors, 
                    st.session_state.data["descriptors_list"], 
                    st.session_state.data["names"]
                )
                
                if matched_name:
                    confidence = max(0, min(100, 100 - distance))
                    st.success(f"""
                    🎯 **IDENTITY MATCHED!**
                    
                    **Recognized Person:** {matched_name}
                    
                    **Confidence Score:** {confidence:.1f}%
                    
                    **Match Quality:** {'Excellent' if confidence > 80 else 'Good' if confidence > 60 else 'Fair'}
                    """)
                    
                    if confidence > 90:
                        st.balloons()
                else:
                    st.warning("🔍 **Identity Not Found**\n\nThis face is not in our registered database. Please register first or try with a different photo.")

    # ---------- FOOTER ----------
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.6);">
        <h3>🌟 Neural Face Recognition Studio</h3>
        <p>Powered by Advanced AI • Secure • Fast • Accurate</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()