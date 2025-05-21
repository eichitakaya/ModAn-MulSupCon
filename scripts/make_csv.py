"""
miniRINのパスと対応するマルチラベルをcsvファイルにまとめる
filepath, label1, label2, label3を列とし、画像ファイル名を行とするcsvファイルを作成する
label1はモダリティ(3), label2は部位(11), label3は疾患名またはnormal（165）

"""

import os
import pandas as pd
import glob

# モダリティと部位の辞書
modality_dir = {"CT": 0, "MR": 1, "US": 2}
region_dir = {"abd": 0, "lung": 1, "af": 2, "brain": 3, "hip": 4, "knee": 5, "shoulder": 6, "spine": 7, "thyroid": 8}

# miniRINのパス
miniRIN_path = "../../data/miniRIN"


# ヘッダー
header = ["filepath", "label1", "label2"]

# pathのリストを取得
paths = glob.glob("../../data/miniRIN/**/*.png", recursive=True)


# pathを確認し、辞書からラベル情報を取得してmodality, regionの順に並べる
multi_labels = []
for path in paths:
    # ../../data/miniRIN/の次がモダリティ
    modality = path.split("/")[4]
    # 辞書からモダリティのラベルを取得
    modality_label = modality_dir[modality]
    # ../../data/miniRIN/の次の次が部位
    region = path.split("/")[5]
    # 辞書から部位のラベルを取得
    region_label = region_dir[region]
    # マルチラベル情報を取得
    multi_labels.append([path, modality_label, region_label])

#print(multi_labels[:10])
# マルチラベル情報をcsvファイルに保存
multi_labels_df = pd.DataFrame(multi_labels, columns=header)
multi_labels_df.to_csv("../../data/miniRIN/multilabel.csv", index=False)

