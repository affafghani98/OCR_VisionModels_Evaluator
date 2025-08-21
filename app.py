import os, io, base64, logging
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import pytesseract
import easyocr
from groq import Groq
import requests, traceback

# ---------------- Setup ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Set GROQ_API_KEY in .env")
    st.stop()

groq = Groq(api_key=GROQ_API_KEY)
easy_reader = easyocr.Reader(["en"], gpu=False)

# Strict OCR prompt for Groq models
VISION_PROMPT_GROQ = (
    "You are an OCR assistant. "
    "Extract **all visible text** from the image exactly as shown. "
    "Do not summarize. Do not explain. "
    "If the image contains tables, preserve the structure using spaces or tabs. "
    "If the image contains diagrams, graphs, or charts with text, extract all labels and numbers. "
    "Return only the raw extracted text."
)

# More general prompt for Ollama local models
VISION_PROMPT_OLLAMA = (
    "You are a vision-language assistant. "
    "Extract as much text as you can from the image, including tables, labels, and numbers. "
    "Return the text in a clear format. "
    "If you cannot extract text from certain areas, skip them gracefully."
)


# ---------------- Engines ----------------
ENGINE_CHOICES = {
    "Tesseract (OCR)": ("ocr", "tesseract"),
    "EasyOCR (OCR)": ("ocr", "easyocr"),

    # Groq-hosted models
    "LLaMA 4 Scout (Vision)": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "LLaMA 4 Maverick (Vision)": ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"),

    # Local Ollama models
    "LLaVA 7B (Local)": ("local_llama", "llava:7b"),
    "Gemma 3 (4B, Local)": ("local_llama", "gemma3:4b"),
    "Granite 3.2 Vision (2B, Local)": ("local_llama", "granite3.2-vision:2b"),
}

def process_with_engine(img_bytes, engine):
    kind, name = ENGINE_CHOICES[engine]
    
    if kind == "ocr":
        # OCR path
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = ImageEnhance.Contrast(img.convert("L")).enhance(2.0)
        return (
            pytesseract.image_to_string(img)
            if name == "tesseract"
            else " ".join(easy_reader.readtext(np.array(img), detail=0, paragraph=True))
        )

    elif kind == "groq":
        # Groq vision models use strict OCR prompt
        uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        resp = groq.chat.completions.create(
            model=name,
            messages=[
                {"role": "system", "content": VISION_PROMPT_GROQ},
                {"role": "user", "content": [
                    {"type": "text", "text": VISION_PROMPT_GROQ},
                    {"type": "image_url", "image_url": {"url": uri}}  
                ]},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    
    elif kind == "local_llama":
        # Ollama local models use the alternative prompt
        return send_to_local_llama(img_bytes, name, VISION_PROMPT_OLLAMA)


def send_to_local_llama(img_bytes: bytes, model_name: str, prompt: str = VISION_PROMPT_OLLAMA) -> str:
    try:
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        resp = requests.post(f"{OLLAMA_SERVER_URL}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "⚠️ No response received from model.")
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error contacting local LLaMA: {e}"


# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")
st.title("Vision + OCR Model Comparator")

# 1. Upload
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# 2. Selectors
col1, col2 = st.columns(2)
with col1:
    eng1 = st.selectbox("Select model (Left)", list(ENGINE_CHOICES.keys()), key="eng1")
with col2:
    eng2 = st.selectbox("Select model (Right)", list(ENGINE_CHOICES.keys()), key="eng2")

# 3. Compare button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
clicked = st.button("🔍 Compare Models", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# 4. Results
if clicked:
    if not uploaded:
        st.warning("Please upload a file first.")
    else:
        img_bytes = uploaded.read()
        out1 = process_with_engine(img_bytes, eng1)
        out2 = process_with_engine(img_bytes, eng2)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{eng1} Output")
            st.text_area("", out1, height=500)
            st.download_button(
                label=f"⬇️ Download {eng1} Output",
                data=out1,
                file_name=f"{eng1.replace(' ', '_').lower()}_output.txt",
                mime="text/plain"
            )

        with col2:
            st.subheader(f"{eng2} Output")
            st.text_area("", out2, height=500)
            st.download_button(
                label=f"⬇️ Download {eng2} Output",
                data=out2,
                file_name=f"{eng2.replace(' ', '_').lower()}_output.txt",
                mime="text/plain"
            )
