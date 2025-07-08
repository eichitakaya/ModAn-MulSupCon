# pytorchのデータセットクラスを定義するファイル

# 各種ライブラリのインポート
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image




class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class miniRINmultilabel(Dataset):
    """
    miniRINのデータセットクラス
    全てをメモリに置くことはせず、パスのリストだけ持っておく。
    ミニバッチのサンプリング時は、パスのリストからランダムにパスを選択する。
    """
    def __init__(self, transform=None, one_hot=True):
        # csvファイルを読み込む
        self.df = pd.read_csv("../../data/miniRIN/multilabel.csv")
        # パスのリストを取得
        self.paths = self.df["filepath"].tolist()
        # 画像の変換を定義
        self.transform = transform

    def one_hot(self, target):
        a, b = target
        # int64に変換してワンホット化
        a_one_hot = torch.nn.functional.one_hot(a.long(), num_classes=3)
        b_one_hot = torch.nn.functional.one_hot(b.long(), num_classes=9)

        one_hot_target = torch.cat([a_one_hot, b_one_hot], dim=-1)

        return one_hot_target
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        filename = self.paths[idx]
        image = Image.open(filename).convert('L')
        # filenameに対応するラベルをdfから取ってくる
        labels = self.df.loc[self.df["filepath"] == filename].drop("filepath", axis=1).values[0]
        target = torch.FloatTensor(labels) 
        if self.one_hot:
            target = self.one_hot(target)

        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target

class RadImageNetDataset(Dataset):
    """
    RadImageNetのデータセットクラス
    全てをメモリに置くことはせず、パスのリストだけ持っておく。
    ミニバッチのサンプリング時は、パスのリストからランダムにパスを選択する。
    ラベルは165個（../../data/miniRIN/**/**/**のフォルダ名がラベル）
    """
    def __init__(self, transform=None):
        # csvファイルを読み込む
        self.df = pd.read_csv("../../data/miniRIN/multilabel.csv")
        # パスのリストを取得
        self.paths = self.df["filepath"].tolist()
        # ラベルの辞書を作成（{"marrow_inflammation": 164}みたいな感じ）
        self.labels_dict = make_label_dict(self.df)
        #print(self.labels_dict)
        # 画像の変換を定義
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        filename = self.paths[idx]
        image = Image.open(filename).convert('L')

        # filenameに対応するfilepathをdfから取ってくる
        filepath = self.df.loc[self.df["filepath"] == filename].iloc[0]["filepath"]
        # filepathのフォルダ名がラベル
        labels = filepath.split("/")[4:7]
        labels = "/".join(labels)
        labels = self.labels_dict[labels]

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


def make_label_dict(df):
    label_list = []
    for i, row in df.iterrows():
        filepath = row["filepath"]
        #print(filepath)
        labels = filepath.split("/")[4:7]
        labels = "/".join(labels)
        #print(labels)
        label_list.append(labels)
    labels_set = set(label_list)
    labels_dict = {label: i for i, label in enumerate(labels_set)}
    return labels_dict

if __name__ == "__main__":
    # dataset = miniRINmultilabel(transform=TwoCropTransform(transforms.Compose([
    #                             transforms.RandomResizedCrop(size=224),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.RandomApply([
    #                                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #                             ], p=0.8),
    #                             transforms.RandomGrayscale(p=0.2),
    #                             transforms.ToTensor(),
    #                             ])))
    # # データを1枚だけ取り出す
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    # for image, target in dataloader:
    #     print(image[0].shape)
    #     print(target.shape)
    #     break
    dataset = RadImageNetDataset(transform=transforms.ToTensor())
    # データを1枚だけ取り出す
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    for image, target in dataloader:
        print(image[0].shape, target)
        print(target.shape)
        break