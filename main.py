import argparse

from module.utils import seed_everything, Config
from module.trainer import Trainer
from module.predictor import predict


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=777)
  parser.add_argument('--mode', type=str, default="train")
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--img_size', type=int, default=384)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--model_name', default="resnet_50")
  parser.add_argument('--detail', default="v1")
  parser.add_argument('--data_path', default="./data")
  parser.add_argument('--test_img_path', default="./data/train/1.JPG")
  parser.add_argument('--ckpt', type=str, default="./ckpt/resnet_50_v1.pth")
  # parser.add_argument('--clip', default=1)
  # parser.add_argument('--checkpoints', default="microsoft/beit-base-patch16-224-pt22k-ft22k")
  args = parser.parse_args()
  
  CONFIG = Config(**vars(args))
  # CONFIG 저장필요
  seed_everything(CONFIG.seed)
  
  
  if CONFIG.mode == "train":
    trainer = Trainer(CONFIG=CONFIG)
    trainer.setup()
    trainer.train()
  elif CONFIG.mode == "predict":
    # 사진 하나 예측
    predict(config=CONFIG)
  else:
    raise ValueError(f"{CONFIG.mode}is not vaild, you can only use trian or test")
