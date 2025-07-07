# 学習済みのモデルを用いて、下流タスクの評価を行うスクリプト

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import downstream_small_dataset as dd
import argparse
import os

from torchvision.models import resnet18
import torch.backends.cudnn as cudnn

# proxy
os.environ["http_proxy"] = "http://proxy.l2.med.tohoku.ac.jp:8080"
os.environ["https_proxy"] = "http://proxy.l2.med.tohoku.ac.jp:8080"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# modelとdatasetを引数に取り、評価を行う関数
def downstream(saved_epoch_num, dataset, epochs=10, batchsize=32):
    if dataset == "thyroid":
        dataset = dd.ThyroidDataset
    if dataset == "breast":
        dataset = dd.BreastUSDataset
    if dataset == "acl":
        dataset = dd.ACLDataset
    # 5foldのtrain/val/testを行う
    preds_labels_all = []
    for i in range(5):
        # データセットの読み込み
        train_dataset = dataset(i+1, "train", transform=transform)
        val_dataset = dataset(i+1, "val", transform=transform)
        test_dataset = dataset(i+1, "test", transform=transform)
        # データローダーの作成
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
        # Build model
        if args.model_type == "imagenet":
            model = resnet18(pretrained=True)
        elif args.model_type == "scratch":
            model = resnet18(pretrained=False)
        elif args.model_type == "SSL":
            model = resnet18(pretrained=False)
        else:
            raise ValueError("model_type must be 'imagenet', 'scratch' or 'SSL'")
        # 1チャンネル対応に変更
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if args.model_type == "SSL":
            model.load_state_dict(torch.load(f"../experiment/run_0/{saved_epoch_num}_checkpoint.pth.tar")["model_state_dict"])
        
        model = model.cuda()
        cudnn.benchmark = True

        # 学習済みパラメータの読み込み
        model.fc = nn.Identity()

        # 損失関数の定義
        criterion = nn.CrossEntropyLoss()
        # 最適化手法の定義
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        # 学習率のスケジューラーの定義
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # 学習ループ
        # testは、valのlossが最小のモデルを使用
        best_val_loss = 10000
        model.train()

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("epoch: {}, loss: {}".format(epoch+1, running_loss))
            scheduler.step()
            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to("cuda:0")
                    labels = labels.to("cuda:0")
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                print("val_loss: ", val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model

        # test
        # 1である確率をlabelsと結合し、numpy形式でreturn
        test_loss = 0.0
        test_corrects = 0
        preds_fold = []
        labels_fold = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")
                outputs = best_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)
                preds_fold.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                labels_fold.extend(labels.cpu().numpy())
        test_loss = test_loss / len(test_dataset)
        preds_labels_all.append([preds_fold, labels_fold])
        
    return preds_labels_all
 
# コマンドライン引数の処理
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="thyroid")
parser.add_argument("--model_num", type=int)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--model_type", type=str, default="imagenet")
parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run")

args = parser.parse_args()

for exp_num in range(1, args.num_experiments + 1):
    print(f"--- Running Experiment {exp_num}/{args.num_experiments} ---")
    # モデルの評価
    result = downstream(saved_epoch_num=args.model_num, dataset=args.dataset, epochs=args.epochs, batchsize=args.batchsize)
    #print(result)
    # 評価結果の集計と保存
    # result自体を5つのcsvに保存
    # acc, auc, sensitivity, specificityを計算し、csvに保存
    # roc_curveを各foldについて描画、平均のROC曲線も重ねて描画
    # 保存先はmodel_pathと同じディレクトリに{dataset}_result.csv, {dataset}_roc_curve.png

    # resultの保存
    import csv
    import os
    # model_pathからmodel.pthを取り除いたものを取得
    model_path = f"../results/{args.dataset}/miniRIN_HML_{args.model_type}_{args.model_num}/{exp_num}"
    # foldごとの結果を保存
    # dataset名のフォルダを作成
    os.makedirs(model_path + f"/{args.dataset}", exist_ok=True)
    print("make directory: ", model_path + f"/{args.dataset}")
    for i in range(5):
        with open("{}/{}/fold_{}.csv".format(model_path, args.dataset, i+1), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["preds", "labels"])
            # 1である確率とlabelsを保存
            for j in range(len(result[i][0])):
                writer.writerow([result[i][0][j], result[i][1][j]])

    # 各foldでのacc, auc, sensitivity, specificityを計算
    import sklearn.metrics as metrics
    accs = []
    aucs = []
    sensitivities = []
    specificities = []

    for i in range(5):
        preds = result[i][0]
        labels = result[i][1]
        preds = np.array(preds)
        print(preds.shape)
        labels = np.array(labels)
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        auc = metrics.auc(fpr, tpr)
        acc = metrics.accuracy_score(labels, np.where(preds > 0.5, 1, 0))
        tn, fp, fn, tp = metrics.confusion_matrix(labels, np.where(preds > 0.5, 1, 0)).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accs.append(acc)
        aucs.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # 平均のacc, auc, sensitivity, specificityを計算
    acc_mean = np.mean(accs)
    auc_mean = np.mean(aucs)
    sensitivity_mean = np.mean(sensitivities)
    specificity_mean = np.mean(specificities)

    # accs, aucs, sensitivities, specificitiesおよびそれらの平均をcsvに保存
    with open("{}/{}/stats.csv".format(model_path, args.dataset), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["acc", "auc", "sensitivity", "specificity"])
        for i in range(5):
            writer.writerow([accs[i], aucs[i], sensitivities[i], specificities[i]])
        writer.writerow([acc_mean, auc_mean, sensitivity_mean, specificity_mean])

    # roc_curveを5つのfoldについて1つの図に描画。平均のROC曲線も描画。
    plt.figure()
    for i in range(5):
        preds = result[i][0]
        labels = result[i][1]
        preds = np.array(preds)
        labels = np.array(labels)
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        auc = metrics.auc(fpr, tpr)
        # 色は全て薄い灰色
        plt.plot(fpr, tpr, color="lightgray")
    # 平均のROC曲線を描画
    preds_all = []
    labels_all = []
    for i in range(5):
        preds_all.extend(result[i][0])
        labels_all.extend(result[i][1])
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    fpr, tpr, thresholds = metrics.roc_curve(labels_all, preds_all)
    auc = metrics.auc(fpr, tpr)
    # 色は青
    plt.plot(fpr, tpr, label="mean (area = %.2f)" % auc, linestyle="--", color="blue")

    plt.legend()
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.savefig("{}/{}/roc_curve.png".format(model_path, args.dataset))
    plt.close()