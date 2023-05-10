import os
import time
import torch
import einops
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.onnx as onnx
import torchvision.models as models
from natsort import natsorted
from tensorboardX import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data import *
from utils import warp, update_train_log, write_train_log, update_eval_log, write_eval_log, print_eval_log

def main():
  device = 'cpu'

  # DATASET
  images = ["test.jpg"]

  # MODEL
  model_stn = models.resnet50()
  model_stn.fc = nn.Linear(2048, 8)
  model = models.resnet50()
  model.fc = nn.Linear(2048, 720)
  resume_path = './models/main.pth'
  stn_resume_path = './models/stn.pth'
  model.load_state_dict(torch.load(resume_path, map_location=torch.device('cpu')))
  model_stn.load_state_dict(torch.load(stn_resume_path, map_location=torch.device('cpu')))
  model_stn.to(device)
  model.to(device)

  with torch.no_grad():
    model.eval()
    model_stn.eval()

    #MODEL
    img = cv2.imread(os.path.join("./data/demo", images[0]))
    img = cv2.resize(img, (224, 224))/255
    img = einops.rearrange(img, 'h w c -> c h w')
    img = torch.Tensor(img)
    img = img.float().to(device)
    img = torch.unsqueeze(img, 0)
    pred = model(img)

    #top 3 predictions
    max_pred = torch.argsort(pred, dim=1, descending=True)
    max_h = max_pred[0] // 60
    max_m = max_pred[0] % 60

    print(f"[{images[0]}] is roughly {max_h[0].cpu().numpy()}:{max_m[0].cpu().numpy():0>2} ({max_pred[0][0].cpu().numpy()}).\n")


if __name__ == "__main__":
  main()