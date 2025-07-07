"""
HR Assistant with Face Authentication - Complete Single Script
Live Camera Authentication + Multi-Agent Workflow using DeepFace
"""

import streamlit as st
st.set_page_config(layout="wide", page_title="Multi-Agent HR Assistant with Face Auth")

import io
import numpy as np
from PIL import Image
import tempfile
import os
import time
import json
import re

# --- Initialize Session State FIRST ---
# This ensures every key we might access later already exists.
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.environ.get('GROQ_API_KEY', "")
    if not st.session_state.groq_api_key:
        try:
            with open("api.txt", "r") as f:
                st.session_state.groq_api_key = f.read().strip()
        except FileNotFoundError: pass

if "last_agent_output" not in st.session_state: st.session_state.last_agent_output = ""
if "selected_agent" not in st.session_state: st.session_state.selected_agent = "Candidate Screener"
if "cv_text" not in st.session_state: st.session_state.cv_text = ""
if "cv_pdf_processed_filename" not in st.session_state: st.session_state.cv_pdf_processed_filename = None
if "cv_face_path" not in st.session_state: st.session_state.cv_face_path = None
if "candidate_authenticated" not in st.session_state: st.session_state.candidate_authenticated = False
if "req_cv_similarity" not in st.session_state: st.session_state.req_cv_similarity = 0.0


# --- Import packages with error handling ---
try: from groq import Groq
except ImportError: st.error("Lib 'groq' not installed. Run: pip install groq"); st.stop()
try: from deepface import DeepFace
except ImportError: st.error("Lib 'deepface' not installed. Run: pip install deepface"); st.stop()
try: from sentence_transformers import SentenceTransformer; SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError: SENTENCE_TRANSFORMERS_AVAILABLE = False
try: import cv2, av
except ImportError: st.error("Libs OpenCV/PyAV not installed. Run: pip install opencv-python-headless av"); st.stop()
try: from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration; WEBRTC_AVAILABLE = True
except ImportError: st.error("Lib 'streamlit-webrtc' not installed. Run: pip install streamlit-webrtc"); st.stop()
try: import fitz
except ImportError: st.error("Lib PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"); st.stop()


# --- Helper Functions ---

@st.cache_resource
def load_embedding_model():
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try: return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e: st.warning(f"Could not load SentenceTransformer model: {e}")
    return None

def extract_text_from_pdf(pdf_bytes):
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text("text") for page in doc)
        return text.strip() or None
    except Exception as e: st.error(f"PyMuPDF extraction failed: {e}"); return None

def compute_similarity(text1, text2):
    model = load_embedding_model()
    if model:
        try:
            from sentence_transformers.util import cos_sim
            e1, e2 = model.encode(text1, convert_to_tensor=True), model.encode(text2, convert_to_tensor=True)
            return cos_sim(e1, e2).item()
        except Exception: pass
    words1, words2 = set(text1.lower().split()), set(text2.lower().split())
    intersection, union = len(words1.intersection(words2)), len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

def extract_face_from_pdf_deepface(pdf_bytes):
    st.info("🔍 Searching for a face in the CV PDF...")
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                for img in page.get_images(full=True):
                    try:
                        base_image = doc.extract_image(img[0])
                        img_np = cv2.imdecode(np.frombuffer(base_image["image"], np.uint8), cv2.IMREAD_COLOR)
                        # Use a more robust detector for extraction as well
                        extracted_faces = DeepFace.extract_faces(img_np, enforce_detection=True, detector_backend='mtcnn')
                        if extracted_faces:
                            face_img_bgr = extracted_faces[0]['face']
                            face_img_rgb = cv2.cvtColor((face_img_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                                Image.fromarray(face_img_rgb).save(tf, format="PNG")
                                st.success(f"✅ Face detected on page {page.number + 1}!")
                                return tf.name
                    except (ValueError, cv2.error, fitz.fitz.FitzError): continue
    except Exception as e: st.error(f"Error during face extraction: {e}")
    st.warning("⚠️ No face was detected in the CV PDF."); return None

def authenticate_face_photo_upload(candidate_image_pil):
    if not st.session_state.cv_face_path: return False, "No CV face available."
    try:
        result = DeepFace.verify(np.array(candidate_image_pil.convert('RGB')), st.session_state.cv_face_path, enforce_detection=True, detector_backend='mtcnn')
        return (True, "Face match! Auth successful.") if result["verified"] else (False, "Face does not match.")
    except ValueError: return False, "Could not detect a face in the uploaded photo."
    except Exception as e: return False, f"Auth error: {e}"

def get_groq_response(api_key, agent_name, inputs):
    if not api_key: st.error("Groq API key is missing."); return None
    try:
        client = Groq(api_key=api_key)
        sys_prompt = AGENT_SYSTEM_PROMPTS[agent_name]["prompt"]
        user_prompt = "\n\n".join(f"{k}:\n{v}" for k, v in inputs.items())
        completion = client.chat.completions.create(messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], model="llama3-8b-8192")
        return completion.choices[0].message.content
    except Exception as e: st.error(f"Groq API Error: {e}"); return None

# --- NEW: A MORE ROBUST VIDEO PROCESSOR ---
class RobustDeepFaceAuthProcessor(VideoProcessorBase):
    def __init__(self):
        self.is_verified = False
        self.auth_message = "Initializing..."
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # If already verified, do nothing further to save resources
        if self.is_verified:
            img_with_text = frame.to_ndarray(format="bgr24")
            cv2.putText(img_with_text, self.auth_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img_with_text, format="bgr24")

        img = frame.to_ndarray(format="bgr24")
        cv_face_path = st.session_state.get("cv_face_path", None)

        if not cv_face_path:
            self.auth_message = "ERROR: Process a CV with a face first"
        elif self.frame_count % 15 == 0: # Process less frequently to handle slower model
            try:
                # KEY CHANGE: Using 'mtcnn' detector for better accuracy in live video
                # silent=True suppresses console logs from deepface
                result = DeepFace.verify(
                    img,
                    cv_face_path,
                    enforce_detection=True,
                    detector_backend='mtcnn',
                    silent=True
                )
                if result["verified"]:
                    self.auth_message = "✅ Match Confirmed! Please wait..."
                    self.is_verified = True
                    # Set the main app's state. The main app will handle the rerun.
                    st.session_state.candidate_authenticated = True
                else:
                    self.auth_message = "⚠️ Face does not match CV"
            except ValueError:
                # This is crucial feedback: tells you if a face wasn't found
                self.auth_message = "No face detected in camera"
        
        self.frame_count += 1
        
        color = (0, 255, 0) if self.is_verified else (0, 0, 255)
        cv2.putText(img, self.auth_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Agent Definitions ---
AGENT_SYSTEM_PROMPTS = { "Job Description Writer": {"prompt": "...", "inputs": ["Job Title", "Key Responsibilities/Skills"], "output_description": "..."}, "Candidate Screener": {"prompt": "...", "inputs": ["Job Description", "Candidate Resume Content"], "output_description": "..."}, "CV-to-Requirements Matcher": {"prompt": "...", "inputs": ["Job Requirements", "Candidate CV Content"], "output_description": "..."}, "Interview Question Generator": {"prompt": "You are an HR expert... generate 5 multiple-choice (QCM) questions... Format exactly like this:\nQ: ...\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: X\n", "inputs": ["Job Requirements"], "output_description": "Five QCM interview questions."}}

# --- UI Rendering ---
st.title("🚀 Multi-Agent HR Assistant with Face Authentication")

with st.sidebar:
    st.header("🔧 Configuration")
    st.session_state.groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key)
    st.markdown("---")
    st.session_state.selected_agent = st.selectbox("Select HR Agent:", list(AGENT_SYSTEM_PROMPTS.keys()))
    st.markdown("---")
    st.subheader("📄 Candidate CV")
    uploaded_cv_pdf = st.file_uploader("Upload CV (PDF)", type="pdf", key="cv_uploader")

    if uploaded_cv_pdf:
        if st.button("🔍 Process CV PDF", use_container_width=True):
            with st.spinner("Processing CV..."):
                st.session_state.cv_pdf_processed_filename = uploaded_cv_pdf.name
                st.session_state.cv_text = extract_text_from_pdf(uploaded_cv_pdf.getvalue())
                st.session_state.cv_face_path = extract_face_from_pdf_deepface(uploaded_cv_pdf.getvalue())
                st.session_state.candidate_authenticated = False
            st.rerun()

    if st.session_state.cv_pdf_processed_filename: st.success(f"✅ Using CV: {st.session_state.cv_pdf_processed_filename}")
    if st.session_state.cv_text:
        with st.expander("Preview CV Text"): st.text_area("", st.session_state.cv_text[:500] + "...", height=150, disabled=True)
    if st.session_state.cv_face_path: st.image(st.session_state.cv_face_path, caption="Detected Face from CV", use_column_width=True)

# Main Area
if not st.session_state.groq_api_key: st.warning("Please add your Groq API key in the sidebar to begin."); st.stop()

agent_config = AGENT_SYSTEM_PROMPTS[st.session_state.selected_agent]
st.header(f"🤖 Agent: {st.session_state.selected_agent}")
st.markdown(f"**🎯 Goal:** {agent_config['output_description']}")

# --- Authentication Gate ---
show_auth_gate = (st.session_state.selected_agent == "Interview Question Generator" and not st.session_state.candidate_authenticated)

if show_auth_gate:
    if not st.session_state.cv_face_path:
        st.error("❌ A CV with a detectable face must be processed before using this agent.")
    else:
        st.error("🔒 **FACE AUTHENTICATION REQUIRED**"); st.info("To access interview questions, please verify your identity.")
        auth_tab1, auth_tab2 = st.tabs(["📸 Upload Photo", "🎥 Live Webcam"])
        with auth_tab1:
            candidate_photo = st.file_uploader("Upload a clear photo", type=["jpg", "png"], key="photo_uploader")
            if candidate_photo and st.button("🔍 Authenticate Photo", type="primary"):
                with st.spinner("Verifying..."):
                    is_match, message = authenticate_face_photo_upload(Image.open(candidate_photo))
                    if is_match: st.session_state.candidate_authenticated = True; st.success(f"✅ AUTHENTICATED! {message}"); st.balloons(); time.sleep(2); st.rerun()
                    else: st.error(f"❌ FAILED! {message}")
        with auth_tab2:
            st.info("The camera will start automatically. Position your face in the center.")
            ctx = webrtc_streamer(key="deepface_webrtc_robust", video_processor_factory=RobustDeepFaceAuthProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, media_stream_constraints={"video": True, "audio": False})
            
            # Check the state set by the processor
            if st.session_state.candidate_authenticated:
                # A short delay can help ensure the UI has time to register the success message before rerun
                time.sleep(1)
                st.rerun()
else:
    # --- Agent UI (if not blocked by auth) ---
    inputs = {}
    for key in agent_config["inputs"]:
        if "Content" in key and st.session_state.cv_text:
            st.info("✓ CV Content automatically loaded."); inputs[key] = st.session_state.cv_text
        else: inputs[key] = st.text_area(f"📝 Input: {key}", key=f"{st.session_state.selected_agent}_{key}", height=150)

    if st.button(f"🚀 Run {st.session_state.selected_agent}", type="primary", use_container_width=True):
        if all(v.strip() for v in inputs.values()):
            with st.spinner(f"🤖 {st.session_state.selected_agent} is thinking..."):
                st.session_state.last_agent_output = get_groq_response(st.session_state.groq_api_key, st.session_state.selected_agent, inputs)
                if st.session_state.selected_agent == "CV-to-Requirements Matcher":
                    st.session_state.req_cv_similarity = compute_similarity(inputs["Job Requirements"], st.session_state.cv_text)
                st.rerun()
        else: st.warning("⚠️ Please fill in all input fields.")

# --- Display last agent output ---
if st.session_state.last_agent_output:
    st.markdown("---"); st.subheader("🎯 Agent Output")
    st.markdown(st.session_state.last_agent_output)
    st.download_button("💾 Download Output", st.session_state.last_agent_output, f"{st.session_state.selected_agent.replace(' ', '_')}.txt")