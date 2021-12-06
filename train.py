from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
else:
    # 랜덤 시드 고정
    torch.manual_seed(777)

transform_train = transforms.Compose([ transforms.ToTensor(), ])
data_set = ImageFolder('dataset/train',transform=transform_train )
classes = data_set.classes
print(classes)

exit()


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 32, 32, 3)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 3200 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(3200, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 43, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out

# CNN 모델 정의
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

training_epochs = 1
batch_size = 10

data_loader = torch.utils.data.DataLoader(data_set, batch_size = batch_size, shuffle = True )
total_batch = len(data_loader)
print('total batch count : {}'.format(total_batch))


# for epoch in range(training_epochs):
for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


dirname = './weights/'
if not os.path.exists(dirname):
    os.mkdir(dirname)

torch.save(model, dirname + 'model.pt')  # 전체 모델 저장
torch.save(model.state_dict(), dirname + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
torch.save()
