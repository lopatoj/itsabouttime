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
	model_stn = models.resnet50(pretrained=True).cpu()
	model_stn.fc = nn.Linear(2048, 8)
	model = models.resnet50(pretrained=True).cpu()
	model.fc = nn.Linear(2048, 720)
	resume_path = './models/{}.pth'.format("full+++")
	stn_resume_path = './models/{}_st.pth'.format("full+++")
	model.load_state_dict(torch.load(resume_path, map_location=torch.device('cpu')))
	model_stn.load_state_dict(torch.load(stn_resume_path, map_location=torch.device('cpu')))

	dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

	input_names = ["batch", "channels", "x", "y"]
	output_names = ["time"]

	torch.onnx.export(model, dummy_input, "analog.onnx")
	torch.onnx.export(model_stn, dummy_input, "analog_stn.onnx")

if __name__ == "__main__":
	main()