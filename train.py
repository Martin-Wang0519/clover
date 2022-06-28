import os.path

import torch
import torch.onnx
from torch.autograd import Variable
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
import torchvision.models
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import split_dataset


class Trainer(object):
    def __init__(self, model, dataset_path, model_weight_save_path):
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.0001
        self.weight_decay = 0.00001
        self.gamma = 0.9
        self.num_workers = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.criterion = torch.nn.CrossEntropyLoss()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)

        self.dataset_path = dataset_path
        self.model_weight_save_path = model_weight_save_path
        self.val_loader = None
        self.train_loader = None

        self.train_loss = []
        self.train_acc = []
        self.validate_loss = []
        self.validate_acc = []

    def training(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_data = 0

        self.model.train()
        train_bar = tqdm(self.train_loader)
        for data in train_bar:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            # 梯度清零
            self.optimizer.zero_grad()
            # 正向传播
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += torch.eq(predicted, labels).sum().item()
            # 计算损失
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            # 反向传播
            loss.backward()
            # 权重更新
            self.optimizer.step()

            total_data += labels.size(0)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} ".format(epoch + 1,
                                                                      self.epochs,
                                                                      loss)
        # 更新学习率
        self.scheduler.step()

        loss = total_loss / len(self.train_loader)
        acc = 100 * total_correct / total_data
        self.train_loss.append(loss)
        self.train_acc.append(acc)

        print('accuracy on train set:%d %%' % acc)

    # 验证函数
    def validating(self, epoch):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_data = 0
        with torch.no_grad():
            val_bar = tqdm(self.val_loader)
            for data in val_bar:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                # 正向传播
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total_correct += torch.eq(predicted, labels).sum().item()

                # 计算损失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                total_data += labels.size(0)

                # 进度条描述训练进度
                val_bar.desc = "validate epoch[{}/{}]".format(epoch + 1,
                                                              self.epochs)

            loss = total_loss / len(self.val_loader)
            acc = 100 * total_correct / total_data
            self.validate_loss.append(loss)
            self.validate_acc.append(acc)

            print('accuracy on validate set:%d %%\n' % acc)

    def start(self):

        print("using {} device.".format(self.device))

        split_dataset(self.dataset_path)

        data_transform = {
            "train": transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor()]),
            "val": transforms.Compose([transforms.Resize([224, 224]),
                                       transforms.ToTensor()])}

        train_dataset = datasets.ImageFolder(root=os.path.join(self.dataset_path, "train"),
                                             transform=data_transform['train'])
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size,
                                       num_workers=self.num_workers)

        val_dataset = datasets.ImageFolder(root=os.path.join(self.dataset_path, "val"), transform=data_transform['val'])
        self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size,
                                     num_workers=self.num_workers)

        for epoch in range(self.epochs):
            self.training(epoch)
            self.validating(epoch)

        torch.save(self.model.state_dict(), self.model_weight_save_path)
        epoch = np.arange(1, self.epochs + 1)
        dataframe = pd.DataFrame({'epoch': epoch,
                                  'train loss': self.train_loss,
                                  'train accuracy': self.train_acc,
                                  'validate loss': self.validate_loss,
                                  'validate accuracy': self.validate_acc
                                  })
        dataframe.to_csv(r"model_data/training_process.csv")


if __name__ == "__main__":
    a = Trainer('dataset/stock_data', 'model_data/model_weight_cuda.pth')
    a.start()
