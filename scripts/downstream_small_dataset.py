
# thyroid分類問題のDownstreamタスクを実行する

# ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


# datesetclassの定義
# thyroid
class ThyroidDataset(torch.utils.data.Dataset):
    # foldとtrainvaltestによって読み込むcsvを変更
    def __init__(self, fold, trainvaltest, transform=None):
        self.dataset_path = "../downstream_data/thyroid/"
        self.transform = transform
        # trainvaltestによって読み込むcsvを変更
        if trainvaltest == "train":
            self.csv_path = self.dataset_path + "dataframe/thyroid_train_fold" + str(fold) + ".csv"
        elif trainvaltest == "val":
            self.csv_path = self.dataset_path + "dataframe/thyroid_val_fold" + str(fold) + ".csv"
        elif trainvaltest == "test":
            self.csv_path = self.dataset_path + "dataframe/thyroid_test_fold" + str(fold) + ".csv"
        else:
            raise ValueError("trainvaltest must be 'train', 'val', or 'test'")
        # csvのpathリストを見て画像を読み込み(Pillowを使用。ヘッダーは無視)
        self.imgs = []
        self.labels = []
        with open(self.csv_path, "r") as f:
            f.readline()
            for line in f:
                # 画像のpathとラベルを取得. ラベルはbenign:0, malignant:1に変換
                img_path, label = line.split(",")
                self.imgs.append(Image.open(self.dataset_path + img_path.strip()))
                label = label.strip()
                if label == "benign":
                    self.labels.append(0)
                elif label == "malignant":
                    self.labels.append(1)
                else:
                    raise ValueError("label must be 'benign' or 'malignant'")


        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# breast US
class BreastUSDataset(torch.utils.data.Dataset):
    # foldとtrainvaltestによって読み込むcsvを変更
    def __init__(self, fold, trainvaltest, transform=None):
        self.dataset_path = "../downstream_data/breast/"
        self.transform = transform
        # trainvaltestによって読み込むcsvを変更
        if trainvaltest == "train":
            self.csv_path = self.dataset_path + "dataframe/breast_train_fold" + str(fold) + ".csv"
        elif trainvaltest == "val":
            self.csv_path = self.dataset_path + "dataframe/breast_val_fold" + str(fold) + ".csv"
        elif trainvaltest == "test":
            self.csv_path = self.dataset_path + "dataframe/breast_test_fold" + str(fold) + ".csv"
        else:
            raise ValueError("trainvaltest must be 'train', 'val', or 'test'")
        # csvのpathリストを見て画像を読み込み(Pillowを使用。ヘッダーは無視)
        self.imgs = []
        self.labels = []
        with open(self.csv_path, "r") as f:
            f.readline()
            for line in f:
                # 画像のpathとラベルを取得. ラベルはbenign:0, malignant:1に変換
                img_path, label = line.split(",")
                self.imgs.append(Image.open(self.dataset_path + img_path.strip()))
                label = label.strip()
                if label == "benign":
                    self.labels.append(0)
                elif label == "malignant":
                    self.labels.append(1)
                else:
                    raise ValueError("label must be 'benign' or 'malignant'")


        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# acl
class ACLDataset:
    def __init__(self, fold, trainvaltest, transform=None):
        self.dataset_path = "../downstream_data/acl/"
        self.transform = transform
        # trainvaltestによって読み込むcsvを変更
        if trainvaltest == "train":
            self.csv_path = self.dataset_path + "dataframe/train_fold" + str(fold) + ".csv"
        elif trainvaltest == "val":
            self.csv_path = self.dataset_path + "dataframe/val_fold" + str(fold) + ".csv"
        elif trainvaltest == "test":
            self.csv_path = self.dataset_path + "dataframe/test_fold" + str(fold) + ".csv"
        else:
            raise ValueError("trainvaltest must be 'train', 'val', or 'test'")
        # csvのpathリストを見て画像を読み込み(Pillowを使用。ヘッダーは無視)
        self.imgs = []
        self.labels = []
        with open(self.csv_path, "r") as f:
            f.readline()
            for line in f:
                # 画像のpathとラベルを取得. ラベルはacl:0, meniscus:1に変換
                img_path, label = line.split(",")
                self.imgs.append(Image.open(self.dataset_path + img_path.strip()))
                label = label.strip()
                if label == "yes":
                    self.labels.append(0)
                elif label == "no":
                    self.labels.append(1)
                else:
                    raise ValueError("label must be 'yes' or 'no'")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 

# データセット読み込みのテスト
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    thyroid = ThyroidDataset(1, trainvaltest="train", transform=transform)
    data_loader = torch.utils.data.DataLoader(thyroid, batch_size=4, shuffle=True)
    for inputs, labels in data_loader:
        print(inputs.size())
        print(labels)
        break
