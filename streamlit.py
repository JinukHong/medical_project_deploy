from glob import glob
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ultralytics import YOLO

from module.utils import crop_preprocess
from module.model import ResNet50

import streamlit as st

import os
import gdown

def download_if_not_exists(file_path, gdrive_id):
    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, file_path, quiet=False)


@st.cache_data
def get_model():
    # 다운로드 경로와 ID 지정
    resnet_path = "ckpt/ResNet50_v0.pth"
    yolo_path = "yolov8n-pose.pt"
    
    download_if_not_exists(resnet_path, "130N1bBrYHrXtJao6Jq2JlQA6RgVS3jgx")
    download_if_not_exists(yolo_path, "1htemHZjg3kYa98tFZusBOGb2CbYPG_YR")

    keypoint_model = YOLO(yolo_path)
    model = ResNet50()
    ckpt = torch.load(resnet_path, map_location='cpu')
    model.load_state_dict(ckpt["model_state_dict"])
    
    return keypoint_model, model

st.title("척추측만증 예측")

img_byte = st.file_uploader("이미지를 선택하세요", type=["jpg", "png", "jpeg"])

if img_byte is not None:
    img = Image.open(img_byte).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = np.array(img)

    st.image(
        img,
        caption=f"image shape: {img.shape[0:2]}",
        use_container_width=True,  # 수정됨
    )

    k_model, model = get_model()

    test_aug = A.Compose([
        A.Normalize(),
        A.Resize(384, 384),
        ToTensorV2()
    ])

    start = time.time()

    crop_img = crop_preprocess(img, k_model)

    # ----- 💡 예외처리: 채널 수 확인 및 보정 -----
    if crop_img is None:
        st.error("이미지 전처리(crop_preprocess) 결과가 None입니다.")
        st.stop()

    if len(crop_img.shape) == 2:  # Grayscale
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
    elif crop_img.shape[2] == 4:  # RGBA
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGBA2RGB)
    elif crop_img.shape[2] != 3:
        st.error(f"알 수 없는 채널 형식입니다: {crop_img.shape}")
        st.stop()

    test_img = test_aug(image=crop_img)["image"].unsqueeze(0)

    model.eval()
    with torch.no_grad():
        result = model(test_img)
        result = result.squeeze(0).cpu().numpy()[0]

    end = time.time()

    st.text(f"추론 시간(cpu): {end - start:.3f} sec")
    st.text(f"척추 측만증일 확률: {result * 100:.3f}%")
