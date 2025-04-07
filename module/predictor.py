import os

import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch.nn as nn

from module.utils import Config, crop_preprocess
from module.model import ResNet50

def predict(config: Config):
  
  img = cv2.imread(config.test_img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  keypoint_model = YOLO("yolov8n-pose.pt")
  
  test_aug = A.Compose([
    A.Normalize(),
    A.Resize(config.img_size, config.img_size),
    ToTensorV2()
  ])
  
  crop_img = crop_preprocess(img, keypoint_model)
  test_img = test_aug(image=crop_img)["image"].unsqueeze(0).to(config.device)
  
  resnet = ResNet50()
  ckpt = torch.load(config.ckpt, map_location=config.device)
  resnet.load_state_dict(ckpt["model_state_dict"])
  weight_fc = resnet.model.fc.weight.data.T
  mod = nn.Sequential(*list(resnet.model.children())[:-2])
  W = torch.unsqueeze(weight_fc, dim=-1).to(config.device)
  
  fig, ax = plt.subplots(figsize=(15, 15))
  
  resnet.to(config.device)
  mod.to(config.device)
  resnet.eval()
  with torch.no_grad():
    feature_map = mod(test_img).squeeze(dim=0)
    result = resnet(test_img)[0][0]
    cam = torch.mul(feature_map, W)
    cam = torch.sum(cam, dim=1) # 채널 모두 합침
    cam = cam.cpu().numpy()
    final_cam = cv2.resize(cam, dsize=(config.img_size, config.img_size), interpolation=cv2.INTER_CUBIC)
    
    crop_img = cv2.resize(crop_img, dsize=(config.img_size, config.img_size), interpolation=cv2.INTER_CUBIC)
    ax.imshow(crop_img)
    ax.imshow(final_cam, alpha=0.4, cmap="jet")
    plt.show()
  
  print(f"척추 측만증일 확률: {result*100:.3f}%")
    
  
  
  