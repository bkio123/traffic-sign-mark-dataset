from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import glob
import cv2
from model import CNN

import pandas as pd
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = CNN()
model = torch.load('weights/model.pt')
print(model)

model.load_state_dict(torch.load('weights/model_state_dict.pt'))
model.to(device)
model.eval()

files = glob.glob('dataset/train/01/*.png')
print(len(files))

file = files[1]
print('file name : ', file)
img = cv2.imread(file)

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.array( img / 255, dtype=np.float32)
    img = np.transpose(img, (2,0,1))
    
    in_img = torch.from_numpy(img)
    in_img = in_img.unsqueeze(dim=0)
    in_img = in_img.to(device)
    return in_img

for i in range(10):

    file = files[i]
    img = cv2.imread(file)

    in_img = preprocess(img)
    
    t1 = time.time()
    prediction = model(in_img)
    out = torch.argmax(prediction)
    print('result : ', out.detach().cpu().numpy())
    elaps = time.time() - t1
    print(elaps)


