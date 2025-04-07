from copy import deepcopy
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from clearml import Task, Logger

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ultralytics import YOLO

from module.utils import Config, get_logger
from module.dataset import ScoliosisDataset_v0
from module.model import build_model

class Trainer:
  def __init__(self, CONFIG: Config):
    self.config = CONFIG
    self.keypoint_model = YOLO("yolov8n-pose.pt")
    # self.logger = get_logger(f"{self.config.model_name}_{self.config.detail}.log")
    self.task = Task.init(project_name="scoliosis-predict", task_name=f"{self.config.model_name}_{self.config.detail}_exp")
    # 환경변수 저장 (하이퍼 파라미터 저장)
    # self.task.connect(CONFIG.__dict__)
    
  def setup(self):
    # TODO
    # csv file 경로 환경변수로 수정
    df = pd.read_csv(os.path.join(self.config.data_path, "scoliosis_v1.csv"))
    
    # csv 파일 형식따라 수정 필요
    df["path"] = df["path"].apply(lambda x: os.path.join(self.config.data_path, x))
    # null 값 삭제
    df.dropna(axis=0, inplace=True)
    
    try:
      # self.model = eval(f"{self.config.model_name}()")
      self.model = build_model(self.config.model_name)
    except Exception as e:
      raise NameError(f"{self.config.model_name} is not exist. please check model name")
    
    self.loss_fn = nn.BCELoss()
    
    if self.config.mode == "train":
      x_train, x_val, y_train, y_val = train_test_split(
        df["path"].values,
        df["type"].values,
        test_size=0.2,
        random_state=self.config.seed,
        stratify=df["type"])
      
      train_transform = A.Compose([
        A.Normalize(),
        A.Resize(self.config.img_size, self.config.img_size),
        A.HorizontalFlip(),
        A.OneOf([
          A.RandomBrightnessContrast(),
          A.Blur(),
          A.CoarseDropout()
        ]),
        A.OneOf([
          A.ColorJitter(),
          A.ChannelShuffle()
        ]),
        A.Rotate(limit=5),
        ToTensorV2()
      ])
      
      val_transform = A.Compose([
        A.Normalize(),
        A.Resize(self.config.img_size, self.config.img_size),
        ToTensorV2()
      ])
      
      train_dataset = ScoliosisDataset_v0(img_pahts=x_train, labels=y_train, transforms=train_transform, keypoint_model=self.keypoint_model)
      val_dataset = ScoliosisDataset_v0(img_pahts=x_val, labels=y_val, transforms=val_transform, keypoint_model=self.keypoint_model)
      
      self.train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True
      )
      
      self.val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=16,
        shuffle=False
      )
    
    elif self.config.mode == "test":
      pass
      
  def train(self):
    self.model.to(self.config.device)
    self.loss_fn.to(self.config.device)
    
    optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.lr)
    lr_scheduler = None
    
    early_stop = 0 
    best_val_acc = 0
    best_model = None
    
    
    for epoch in range(1, self.config.epochs+1):
      self.model.train()
      train_loss_lst = []
      for imgs, labels in tqdm(self.train_dataloader):
        imgs = imgs.to(self.config.device)
        labels = labels.float().to(self.config.device)
        
        optimizer.zero_grad()
        output = self.model(imgs)
        output = output.squeeze(-1)
        loss = self.loss_fn(output, labels)
        loss.backward()
        
        optimizer.step()
        
        train_loss_lst.append(loss.item())
      
      val_loss, val_acc = self.valid()
      train_loss = np.mean(train_loss_lst)

      # self.logger.info(f"EPOCH: {epoch} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")
      Logger.current_logger().report_scalar(
        "Loss", "train loss", iteration=epoch, value=train_loss
      )

      Logger.current_logger().report_scalar(
        "Loss", "val loss", iteration=epoch, value=val_loss
      )
      
      Logger.current_logger().report_scalar(
        "Acc", "val acc", iteration=epoch, value=val_acc
      )

      if best_val_acc < val_acc:
        early_stop = 0
        best_val_acc = val_acc
        best_model = deepcopy(self.model)
      
      if early_stop >= 5:
        break
    
    os.makedirs("ckpt", exist_ok=True)
    torch.save({"model_state_dict": best_model.state_dict()}, f"ckpt/{self.config.model_name}_{self.config.detail}.pth")
    
    
  def valid(self):
    self.model.eval()
    with torch.no_grad():
      val_loss = []
      val_acc = []
      for imgs, labels in tqdm(self.val_dataloader):
        imgs = imgs.to(self.config.device)
        labels = labels.float().to(self.config.device)
        
        output = self.model(imgs)
        output = output.squeeze(-1)
        loss = self.loss_fn(output, labels)
        
        output = output.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        preds = output > 0.5
        batch_acc = (labels == preds).mean()
        val_acc.append(batch_acc)
        val_loss.append(loss.item())
    
    return np.mean(val_loss), np.mean(val_acc)
