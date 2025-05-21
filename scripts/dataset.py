# pytorchのデータセットクラスを定義するファイル

# 各種ライブラリのインポート
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


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

    # 画像に対してマルチラベルを付与する関数
    # def get_multilabel(self, image_path):
    #     # 画像パスからファイル名を取得
    #     filename = os.path.basename(image_path)
    #     # マルチラベルのcsvファイルからマルチラベルを取得
    #     return multilabel[multilabel['filename'] == filename]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return
    

if __name__ == "__main__":
    dataset = miniRINmultilabel()
    print(len(dataset))
    print(dataset.paths[0])