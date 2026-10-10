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

col1, col2 = st.columns(2)

with col1:
    archivo = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png", "webp"])
    calidad = st.radio("Resolución / Escala Deseada:", ["2K (2X)", "4K (4X)", "8K (8X)"], index=1)

if archivo is not None:
    imagen_pil = Image.open(archivo).convert('RGB')
    
    with col1:
        st.image(imagen_pil, caption="Imagen Original", use_column_width=True)
        boton_procesar = st.button("✨ Procesar Imagen", type="primary", use_container_width=True)

    if boton_procesar:
        with st.spinner("Reescalando imagen con IA..."):
            factor_escala = {"2K (2X)": 2, "4K (4X)": 4, "8K (8X)": 8}.get(calidad, 4)
            
            ancho, alto = imagen_pil.size
            imagen_reescalada = imagen_pil.resize((ancho * factor_escala, alto * factor_escala), Image.LANCZOS)
            
            with col2:
                st.image(imagen_reescalada, caption=f"Resultado Reescalado {calidad}", use_column_width=True)
                
                out_path = f"resultado_{factor_escala}x.png"
                imagen_reescalada.save(out_path)
                
                with open(out_path, "rb") as file:
                    st.download_button(
                        label="📥 Descargar Resultado",
                        data=file,
                        file_name=f"imagen_reescalada_{calidad}.png",
                        mime="image/png",
                        use_container_width=True
                    )
