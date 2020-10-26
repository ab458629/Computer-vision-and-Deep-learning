'''
main.py
航太所碩一 P46091204 蔡承穎  Copyright (C) 2020
'''
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *
from Ui_HW01_05 import Ui_MainWindow

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary
from tqdm import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

batch_size = 32
learning_rate = 0.001
num_epoches = 20  # Beacause there are not GPU in my PC, I assume epoch is 20.

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download Cifar-10
train_dataset = datasets.CIFAR10(
    './data', train=True, transform=transforms.ToTensor(), download=True)
# Conver trainset to Tensor
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.CIFAR10(
    './data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # 16
            nn.Linear(4096, num_classes),   # output feature 10
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


model = VGG16()
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

# loss function is cross entropy
criterion = nn.CrossEntropyLoss()
# optimzer is Adam (stochastic gradient decent) but we also can use Adgrad SGD and so on
# Adam is combine the advantages of Adagrad and RMSprop
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    # __init__:解構函式，也就是類被建立後就會預先載入的專案。
    # 馬上執行，這個方法可以用來對你的物件做一些你希望的初始化。
    def __init__(self):
        # 這裡需要過載一下mywindow，同時也包含了QtWidgets.QMainWindow的預載入項。
        super(mywindow, self).__init__()
        self.setupUi(self)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()  # torch.FloatTensor -> numpy array
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def button1_clicked():

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    print(' '.join('%5s' % classes[labels[j]] for j in range(10)))
    imshow(torchvision.utils.make_grid(images[:10]))


def button2_clicked():
    print("Hyperparameters:")
    print("Batch size: ", batch_size)
    print("Learning rate: ", learning_rate)
    print("Optimizer: Adam")


def button3_clicked():
    summary(model, (3, 32, 32))  # size of input image is 32x32


def button4_clicked():
    img = cv2.imread('loss and acc.png')
    cv2.imshow('Loss and Acc', img)
    #  If take Inception-V3, ResNet as references, tha accuracy may lift.


def button5_clicked():
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    net = VGG16()
    net.load_state_dict(torch.load('vgg16.pth'))
    net.eval()

    index = window.lineEdit.text()
    index = int(index)

    if (index >= 0 and index <= 9999):
        input = images[index]
        input = input.unsqueeze(0)
        model.eval()
        output = net(input)
        _, pred = torch.max(output.data, 1)

        imshow(torchvision.utils.make_grid(images[index]))

        print('GroundTruth: ', classes[labels[index]])
        print('Predicted: ', classes[pred])

    else:
        print("Please check the type or range of input")


if __name__ == '__main__':  # 如果整個程式是主程式
    app = QtWidgets.QApplication(sys.argv)  # 初始化GUI介面
    window = mywindow()  # 呼叫mywindow物件，並給這個物件名字window
    window.setWindowTitle("HW01_05")
    # 有了例項，就得讓它顯示，show()是QWidget的方法，用於顯示視窗。
    window.show()
    window.pushButton.clicked.connect(button1_clicked)
    window.pushButton_2.clicked.connect(button2_clicked)
    window.pushButton_3.clicked.connect(button3_clicked)
    window.pushButton_4.clicked.connect(button4_clicked)
    window.pushButton_5.clicked.connect(button5_clicked)
    # 呼叫sys庫的exit退出方法，條件是app.exec_()，也就是整個視窗關閉。
    # 有時候退出程式後，sys.exit(app.exec_())會報錯，改用app.exec_()就沒事
    sys.exit(app.exec_())
