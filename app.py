import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

st.set_page_config(page_title="Reescalador IA", layout="wide")

st.title("🚀 Reescalador de Fotos IA 8K (Servidor 24/7)")

@st.cache_resource
def cargar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
    return upsampler

try:
    upsampler = cargar_modelo()
    st.success("Cargado correctamente en la nube.")
except Exception as e:
    st.error(f"Error al iniciar el motor: {e}")

archivo = st.file_uploader("Sube tu imagen aquí", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert('RGB')
    st.image(imagen_pil, caption="Imagen Original", width=400)
    
    if st.button("Procesar Imagen"):
        with st.spinner("Reescalando..."):
            img_np = np.array(imagen_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            output, _ = upsampler.enhance(img_bgr, outscale=4)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            st.image(output_rgb, caption="Resultado Reescalado 4K/8K", use_column_width=True)
            
            res_pil = Image.fromarray(output_rgb)
            res_pil.save("resultado.png")
            
            with open("resultado.png", "rb") as file:
                st.download_button("📥 Descargar Resultado", data=file, file_name="imagen_8k.png", mime="image/png")