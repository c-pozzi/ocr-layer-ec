"""
Simple Streamlit app: upload an image, OCR it with Qwen 7B and 32B, show results side by side.

Usage:
    streamlit run test_ocr_app.py --server.port 8501
"""

import streamlit as st
import base64
import requests

SERVERS = {
    "Qwen 7B AWQ": {"url": "http://localhost:8000/v1/chat/completions", "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"},
    "Qwen 32B AWQ": {"url": "http://localhost:8001/v1/chat/completions", "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"},
}

PROMPT = "Transcribe all the text in this image exactly as written. Preserve formatting, punctuation, and capitalization."


def ocr_request(url: str, model: str, image_b64: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


st.set_page_config(page_title="OCR Test", layout="wide")
st.title("Qwen OCR Comparison")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"])

if uploaded:
    image_bytes = uploaded.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    st.subheader("Source Image")
    st.image(image_bytes, use_container_width=True)

    if st.button("Run OCR", type="primary"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Qwen 7B AWQ")
            with st.spinner("Running 7B..."):
                try:
                    cfg = SERVERS["Qwen 7B AWQ"]
                    result_7b = ocr_request(cfg["url"], cfg["model"], image_b64)
                    st.text_area("Output", result_7b, height=400)
                except requests.ConnectionError:
                    st.error("7B server not running (port 8000)")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.subheader("Qwen 32B AWQ")
            with st.spinner("Running 32B..."):
                try:
                    cfg = SERVERS["Qwen 32B AWQ"]
                    result_32b = ocr_request(cfg["url"], cfg["model"], image_b64)
                    st.text_area("Output", result_32b, height=400)
                except requests.ConnectionError:
                    st.error("32B server not running (port 8001)")
                except Exception as e:
                    st.error(f"Error: {e}")
