# pytorchのデータセットクラスを定義するファイル

# 各種ライブラリのインポート
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class miniRINmultilabel(Dataset):
    """
    miniRINのデータセットクラス
    全てをメモリに置くことはせず、パスのリストだけ持っておく。
    ミニバッチのサンプリング時は、パスのリストからランダムにパスを選択する。
    """
    def __init__(self):
        # csvファイルを読み込む
        self.df = pd.read_csv("../../data/miniRIN/multilabel.csv")
        # パスのリストを取得
        self.paths = self.df["filepath"].tolist()
        # 画像の変換を定義
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        filename = self.paths[idx]
        image = Image.open(filename).convert('RGB')
        # filenameに対応するラベルをdfから取ってくる
        labels = self.df.loc[self.df["filepath"] == filename].drop("filepath", axis=1).values[0]
        target = torch.FloatTensor(labels) 
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    

if __name__ == "__main__":
    dataset = miniRINmultilabel()
    # データを1枚だけ取り出す
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for image, target in dataloader:
        print(image.shape)
        print(target)
        break