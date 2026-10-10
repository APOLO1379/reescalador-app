import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import urllib.request
import os

st.set_page_config(page_title="Reescalador IA 8K", layout="wide")
st.title("🚀 Reescalador de Fotos IA (24/7)")

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
MODEL_PATH = "realesr-general-x4v3.pth"

@st.cache_resource
def cargar_entorno():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    device = cargar_entorno()
    st.success("✅ Servidor listo")
except Exception as e:
    st.error(f"Error: {e}")

archivo = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert('RGB')
    st.image(imagen_pil, caption="Imagen Original", width=350)
    
    if st.button("✨ Procesar Imagen", type="primary"):
        with st.spinner("Reescalando..."):
            ancho, alto = imagen_pil.size
            imagen_reescalada = imagen_pil.resize((ancho * 4, alto * 4), Image.LANCZOS)
            
            st.image(imagen_reescalada, caption="Resultado Reescalado 4X", use_column_width=True)
            
            out_path = "resultado.png"
            imagen_reescalada.save(out_path)
            
            with open(out_path, "rb") as file:
                st.download_button("📥 Descargar Resultado", data=file, file_name="imagen_reescalada.png", mime="image/png")
