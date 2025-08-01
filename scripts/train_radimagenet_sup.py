# resnet18をminiRINを用いて教師あり学習
# ラベルは165個（../../data/miniRIN/**/**/**）

# ライブラリをインポート
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from datasets import RadImageNetDataset
import torch.backends.cudnn as cudnn

# パラメータ
num_epochs = 200
batch_size = 512

# デバイスを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセットのパスを設定
dataset_path = "../../data/miniRIN/**/**/**"

# データセットを読み込む
train_dataset = RadImageNetDataset(transform=transforms.ToTensor())

# データローダーを作成
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# モデルを定義
model = models.resnet18(pretrained=True)
# 1チャンネル対応に変更
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 165)
model = model.cuda()
cudnn.benchmark = True

# モデルをGPUに転送
model = model.to(device)

# 最適化と損失関数を定義
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss {loss.item()}")

# モデルを保存
torch.save(model.state_dict(), "../experiment/radimagenet_sup_200epoch.pth")