# ../experiment/run_0/t_checkpoint.pth.tarのモデルを読み込む（tは1-10）
# ../../data/miniRINの画像に対するベクトルを取得
# UMAPで可視化
# t=1から10まで繰り返し、可視化画像を保存する

import os
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from datasets import miniRINmultilabel
from utils import one_hot2label

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),])

# 1-10まで繰り返し、可視化画像を保存する
for t in range(1, 11):
    # モデルの読み込み
    model_path = f"../experiment/run_0/{t}_checkpoint.pth.tar"
    # resnet18を読み込む
    model = resnet18(weights=None)
    # 1チャンネルに変換
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.fc = torch.nn.Identity()
    model.eval()
    

    # データセットの読み込み
    dataset = miniRINmultilabel(transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 全ての画像に対するベクトルを取得
    vectors = []
    targets = []
    cnt = 0
    for batch in dataloader:
        images = batch[0]
        target = batch[1]
        outputs = model(images)
        vectors.append(outputs[0].detach().cpu().numpy())
        target = one_hot2label(target)
        targets.append(target)
        cnt += 1
    
    # targetsを数値ラベルに変換して辞書に格納
    label_dict = {}
    # targetsの集合を取得
    unique_targets = set(targets)
    for i, target in enumerate(unique_targets):
        label_dict[target] = i
    # targetsを数値ラベルに変換
    targets = [label_dict[target] for target in targets]
    print(unique_targets)
    print(label_dict)

    # UMAPで可視化
    reducer = umap.UMAP(n_components=2)
    embeddings = reducer.fit_transform(vectors)
    cmap = matplotlib.cm.get_cmap('tab20', 11)  # 11色で分割
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=targets, cmap=cmap)
    plt.colorbar()
    plt.savefig(f"../visualize/{t}_umap.png")
    plt.close()

