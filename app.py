import os, io, base64, logging, traceback, time, requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from groq import Groq

# ---------------- Setup ----------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # ensures logs show in backend terminal
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Set GROQ_API_KEY in .env")
    st.stop()

groq = Groq(api_key=GROQ_API_KEY)
easy_reader = easyocr.Reader(["en"], gpu=False)
paddle_reader = PaddleOCR(use_angle_cls=True, lang="en")

DEFAULT_PROMPT = (
    "You are a vision-language assistant. "
    "Extract **all visible text** from the image exactly as shown. "
    "Do not summarize. Do not explain. "
    "If the image contains tables, preserve the structure using spaces or tabs. "
    "If the image contains diagrams, graphs, or charts with text, extract all labels and numbers. "
    "Return the raw extracted text."
)

# ---------------- Engines ----------------
ENGINE_CHOICES = {
    "Tesseract (OCR)": ("ocr", "tesseract"),
    "EasyOCR (OCR)": ("ocr", "easyocr"),
    "PaddleOCR (OCR)": ("ocr", "paddleocr"),

    # Groq-hosted models
    "LLaMA 4 Scout (Vision Model - Groq)": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "LLaMA 4 Maverick (Vision Model - Groq)": ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"),

    # Local Ollama models
    "LLaVA 7B (Local)": ("local_llama", "llava:7b"),
    "Gemma 3 (4B, Local)": ("local_llama", "gemma3:4b"),
    "Granite 3.2 Vision (2B, Local)": ("local_llama", "granite3.2-vision:2b"),
}


def extract_text_from_item(item):
    """Helper function to extract text from various item formats"""
    if isinstance(item, str):
        return item.strip()
    elif isinstance(item, dict):
        text_keys = ['text', 'rec_text', 'recognized_text', 'content', 'transcription']
        for key in text_keys:
            if key in item and item[key]:
                return str(item[key]).strip()
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        # Legacy format [bbox, [text, confidence]]
        if isinstance(item[1], (list, tuple)) and len(item[1]) > 0:
            return str(item[1][0]).strip()
        elif isinstance(item[1], str):
            return item[1].strip()
    return None


def deep_extract_text(obj, max_depth=3, current_depth=0):
    """Recursively extract any text-like content from nested structures"""
    text_lines = []
    
    if current_depth > max_depth:
        return text_lines
    
    if isinstance(obj, str) and len(obj.strip()) > 0 and not obj.startswith('array'):
        text_lines.append(obj.strip())
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and len(value.strip()) > 0 and not value.startswith('array'):
                text_lines.append(value.strip())
            elif isinstance(value, (dict, list)):
                text_lines.extend(deep_extract_text(value, max_depth, current_depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            text_lines.extend(deep_extract_text(item, max_depth, current_depth + 1))
    
    return text_lines


def analyze_result_structure(result, max_depth=2):
    """Analyze the structure of the result for debugging"""
    def analyze_item(obj, depth=0, max_d=2):
        if depth > max_d:
            return "..."
        
        if isinstance(obj, dict):
            keys = list(obj.keys())[:5]  # Show first 5 keys
            return f"dict({keys}...)" if len(obj) > 5 else f"dict({keys})"
        elif isinstance(obj, list):
            if len(obj) > 0:
                return f"list[{len(obj)}]({analyze_item(obj[0], depth+1, max_d)})"
            else:
                return "list[0]"
        elif isinstance(obj, str):
            return f"str('{obj[:30]}...')" if len(obj) > 30 else f"str('{obj}')"
        else:
            return str(type(obj).__name__)
    
    return analyze_item(result)


def process_with_engine(img_bytes, engine, user_prompt):
    kind, name = ENGINE_CHOICES[engine]
    logging.info(f"▶ Using engine: {engine} ({kind} - {name})")
    logging.info(f"▶ Prompt sent: {user_prompt[:150]}....")  # log first 120 chars

    if kind == "ocr":
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = ImageEnhance.Contrast(img.convert("L")).enhance(2.0)
        if name == "tesseract":
            return pytesseract.image_to_string(img)
        elif name == "easyocr":
            return " ".join(easy_reader.readtext(np.array(img), detail=0, paragraph=True))
        elif name == "paddleocr":
            try:
                # Use original RGB image without contrast enhancement for PaddleOCR
                img_original = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img_array = np.array(img_original)
                
                # Ensure image has correct shape (H, W, C)
                if len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, axis=2)
                    img_array = np.repeat(img_array, 3, axis=2)
                elif img_array.shape[2] == 4:  # RGBA to RGB
                    img_array = img_array[:,:,:3]
                
                logging.info(f"PaddleOCR processing image with shape: {img_array.shape}")
                
                # Try both predict and ocr methods
                result = None
                try:
                    result = paddle_reader.predict(img_array)
                    logging.info(f"PaddleOCR predict result type: {type(result)}")
                except Exception as predict_error:
                    logging.warning(f"Predict failed, trying ocr: {predict_error}")
                    try:
                        result = paddle_reader.ocr(img_array)
                        logging.info(f"PaddleOCR ocr result type: {type(result)}")
                    except Exception as ocr_error:
                        logging.error(f"Both predict and ocr failed: {ocr_error}")
                        return f"❌ PaddleOCR API error: {str(ocr_error)}"
                
                if not result:
                    return "No text detected by PaddleOCR (empty result)"
                
                text_lines = []
                
                # Handle the specific PaddleOCR structure we see
                try:
                    logging.info(f"Processing result of type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                    
                    # The result is a list of dictionaries with nested structure
                    for item in result:
                        if isinstance(item, dict):
                            # Look for OCR results in various possible keys
                            ocr_keys_to_check = ['ocr_res', 'text_detection', 'text_recognition', 'predictions', 'results']
                            
                            for key in ocr_keys_to_check:
                                if key in item:
                                    ocr_results = item[key]
                                    logging.info(f"Found OCR results in key '{key}': {type(ocr_results)}")
                                    
                                    if isinstance(ocr_results, list):
                                        for ocr_item in ocr_results:
                                            text_content = extract_text_from_item(ocr_item)
                                            if text_content:
                                                text_lines.append(text_content)
                                                logging.info(f"Extracted text: {text_content}")
                            
                            # Also check if there are direct text fields
                            direct_text_keys = ['text', 'rec_text', 'recognized_text', 'content']
                            for key in direct_text_keys:
                                if key in item and item[key]:
                                    text_content = str(item[key]).strip()
                                    if text_content:
                                        text_lines.append(text_content)
                                        logging.info(f"Found direct text via '{key}': {text_content}")
                            
                            # Check for nested structures
                            if 'doc_preprocessor_res' in item:
                                doc_res = item['doc_preprocessor_res']
                                if isinstance(doc_res, dict):
                                    # Look for text in document preprocessor results
                                    for nested_key in ['text', 'ocr_results', 'recognized_text']:
                                        if nested_key in doc_res and doc_res[nested_key]:
                                            text_content = str(doc_res[nested_key]).strip()
                                            if text_content:
                                                text_lines.append(text_content)
                                                logging.info(f"Found text in doc_preprocessor_res['{nested_key}']: {text_content}")
                    
                    logging.info(f"Total text lines found: {len(text_lines)}")
                    
                    # If no text found with above methods, let's try to extract from any string-like values
                    if not text_lines:
                        logging.info("No text found with standard methods, trying deep extraction...")
                        text_lines = deep_extract_text(result)
                    
                except Exception as parse_error:
                    logging.error(f"Error parsing PaddleOCR result: {parse_error}")
                    return f"PaddleOCR result parsing error: {str(parse_error)}"
                
                if text_lines:
                    final_result = "\n".join(text_lines)
                    logging.info(f"Final PaddleOCR result: {final_result}")
                    return final_result
                else:
                    # Return structure info for debugging
                    structure_info = analyze_result_structure(result)
                    return f"No text extracted. Structure analysis:\n{structure_info}"
                
            except Exception as e:
                logging.error(f"❌ PaddleOCR error: {e}")
                return f"❌ PaddleOCR error: {str(e)}"

    elif kind == "groq":
        uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        resp = groq.chat.completions.create(
            model=name,
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": uri}}
                ]},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()

    elif kind == "local_llama":
        return send_to_local_llama(img_bytes, name, user_prompt)


def send_to_local_llama(img_bytes: bytes, model_name: str, prompt: str) -> str:
    try:
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        logging.info(f"▶ Sending request to local LLaMA model: {model_name}")
        logging.info(f"▶ Prompt: {prompt[:120]}...")
        resp = requests.post(f"{OLLAMA_SERVER_URL}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "⚠️ No response received from model.")
    except Exception as e:
        logging.error("❌ Error contacting local LLaMA", exc_info=True)
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

# 3. Show & edit system prompt
st.subheader("🔧 System Prompt")
user_prompt = st.text_area("You can edit the vision system prompt below:", DEFAULT_PROMPT, height=150)

# 4. Compare button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
clicked = st.button("🔍 Compare Models", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# 5. Results
if clicked:
    if not uploaded:
        st.warning("Please upload a file first.")
    else:
        img_bytes = uploaded.read()

        # --- Measure time for engine 1 ---
        start1 = time.perf_counter()
        out1 = process_with_engine(img_bytes, eng1, user_prompt or DEFAULT_PROMPT)
        end1 = time.perf_counter()
        time1 = end1 - start1

        # --- Measure time for engine 2 ---
        start2 = time.perf_counter()
        out2 = process_with_engine(img_bytes, eng2, user_prompt or DEFAULT_PROMPT)
        end2 = time.perf_counter()
        time2 = end2 - start2

        # --- Show results ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{eng1} Output")
            st.caption(f"⏱️ Processing time: {time1:.2f} seconds")
            st.text_area(f"{eng1} Result", out1, height=500, key=f"out_left_{eng1}")
            st.download_button(
                label=f"⬇️ Download {eng1} Output",
                data=out1,
                file_name=f"{eng1.replace(' ', '_').lower()}_output.txt",
                mime="text/plain",
                key=f"download_left_{eng1}"
            )

        with col2:
            st.subheader(f"{eng2} Output")
            st.caption(f"⏱️ Processing time: {time2:.2f} seconds")
            st.text_area(f"{eng2} Result", out2, height=500, key=f"out_right_{eng2}")
            st.download_button(
                label=f"⬇️ Download {eng2} Output",
                data=out2,
                file_name=f"{eng2.replace(' ', '_').lower()}_output.txt",
                mime="text/plain",
                key=f"download_right_{eng2}"
            )