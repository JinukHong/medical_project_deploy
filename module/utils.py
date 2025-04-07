import os
import random
from dataclasses import dataclass
import logging

import numpy as np

import torch

def seed_everything(seed: int):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  
# def crop_preprocess(img, model):
#   """
#   img를 등부분만 crop해서 반환해주는 함수
#   박스의 width
#   목과 어깨 1:3 지점 엉덩이부분을 height
#   """
#   result = model(img, verbose=False)[0]
  
#   bbox = result.boxes
#   keypoints = result.keypoints
#   # print(bbox)
#   xyxy = bbox.xyxy[0]

#   # 박스 좌표
#   left_x = xyxy[0]
#   right_x = xyxy[2] 

#   t = keypoints.xy[0][0] # 코(머리 부분)dd
#   lt = keypoints.xy[0][5] # 왼쪽 어깨
#   rt = keypoints.xy[0][6] # 오른쪽 어깨
#   lb = keypoints.xy[0][11] # 왼쪽 엉덩이
#   rb = keypoints.xy[0][12] # 오른쪽 엉덩이
  
#   img = img[int((t[1]*3 + min(lt[1], rt[1]))/4):int(max(lb[1], rb[1])), int(left_x):int(right_x)]
#   return img

def crop_preprocess(img, model):
    """
    img를 등부분만 crop해서 반환해주는 함수
    """
    result = model(img, verbose=False)[0]
    
    bbox = result.boxes
    keypoints = result.keypoints

    # 박스 없는 경우 예외 처리
    if bbox is None or len(bbox) == 0:
        print("❌ 탐지된 객체 없음.")
        return None

    # keypoints 없는 경우 예외 처리
    if keypoints is None or keypoints.xy.shape[0] == 0:
        print("❌ keypoints 없음.")
        return None

    try:
        xyxy = bbox.xyxy[0]

        left_x = xyxy[0].item()
        right_x = xyxy[2].item()

        t = keypoints.xy[0][0]  # nose
        lt = keypoints.xy[0][5]  # left shoulder
        rt = keypoints.xy[0][6]  # right shoulder
        lb = keypoints.xy[0][11]  # left hip
        rb = keypoints.xy[0][12]  # right hip

        top = int((t[1] * 3 + min(lt[1], rt[1])) / 4)
        bottom = int(max(lb[1], rb[1]))
        left = int(left_x)
        right = int(right_x)

        # 이미지 크기 초과 방지
        h, w = img.shape[:2]
        top = max(0, min(h, top))
        bottom = max(0, min(h, bottom))
        left = max(0, min(w, left))
        right = max(0, min(w, right))

        cropped = img[top:bottom, left:right]

        if cropped.size == 0:
            print("❌ 잘린 이미지가 비어 있음.")
            return None

        return cropped

    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return None


def get_logger(filename, log_level=logging.INFO):
  os.makedirs("log", exist_ok=True)
  formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
  handler = logging.FileHandler(f"log/{filename}")
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.setLevel(log_level)
  logger.addHandler(handler)
  return logger


@dataclass
class Config:
  seed: int
  mode: str
  epochs: int
  lr: float
  img_size: int
  num_workers: int
  batch_size: int
  model_name: str
  detail: str  
  ckpt: str
  data_path: str
  test_img_path: str
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")