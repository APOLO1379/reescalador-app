import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os

# Parche automático para el error de torchvision/basicsr en Python moderno
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as F
        sys.modules["torchvision.transforms.functional_tensor"] = F
    except Exception:
        pass

st.set_page_config(page_title="Reescalador IA 8K", layout="wide")
st.title("🚀 Reescalador de Fotos IA (24/7)")

@st.cache_resource
def cargar_modelo():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    
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
    st.success("✅ Servidor e IA listos")
except Exception as e:
    st.error(f"Error iniciando modelo: {e}")

archivo = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png", "webp"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert('RGB')
    st.image(imagen_pil, caption="Imagen Original", width=350)
    
    if st.button("✨ Procesar Imagen", type="primary"):
        with st.spinner("Procesando imagen..."):
            img_np = np.array(imagen_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            output, _ = upsampler.enhance(img_bgr, outscale=4)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            st.image(output_rgb, caption="Resultado Reescalado", use_column_width=True)
            
            res_pil = Image.fromarray(output_rgb)
            res_pil.save("resultado.png")
            
            with open("resultado.png", "rb") as file:
                st.download_button("📥 Descargar Resultado", data=file, file_name="imagen_8k.png", mime="image/png")
