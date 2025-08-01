# Linear Probingスクリプト
# エンコーダーの重みを凍結し、分類ヘッドのみを学習する

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
import csv
import sklearn.metrics as metrics

from torchvision.models import resnet18
import torch.backends.cudnn as cudnn

from train_simCLR import SimCLR

# proxy
os.environ["http_proxy"] = "http://proxy.l2.med.tohoku.ac.jp:8080"
os.environ["https_proxy"] = "http://proxy.l2.med.tohoku.ac.jp:8080"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

def load_model(model_type, saved_epoch_num=None):
    """指定されたモデルタイプのモデルを読み込む"""
    if model_type == "imagenet":
        model = resnet18(pretrained=True)
        # 1チャンネル対応に変更
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # エンコーダー部分を凍結
        for param in model.parameters():
            param.requires_grad = False
        # 分類ヘッドのみ学習可能
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    elif model_type == "scratch":
        model = resnet18(pretrained=False)
        # 1チャンネル対応に変更
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # エンコーダー部分を凍結
        for param in model.parameters():
            param.requires_grad = False
        # 分類ヘッドのみ学習可能
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    elif model_type == "SSL":
        model = resnet18(pretrained=False)
        # 1チャンネル対応に変更
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 学習済み重みを読み込み
        model.load_state_dict(torch.load(f"../experiment/run_0/{saved_epoch_num}_checkpoint.pth.tar")["model_state_dict"])
        # エンコーダー部分を凍結
        for param in model.parameters():
            param.requires_grad = False
        # 分類ヘッドのみ学習可能
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    elif model_type == "radimagenet":
        model = resnet18(pretrained=False)
        # 1チャンネル対応に変更
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 165)
        # 学習済み重みを読み込み
        model.load_state_dict(torch.load(f"../experiment/radimagenet_sup_200epoch.pth"))
        # エンコーダー部分を凍結
        for param in model.parameters():
            param.requires_grad = False
        # 分類ヘッドのみ学習可能
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    elif model_type == "simCLR":
        # SimCLRモデルを作成し、学習済み重みを読み込み
        simclr_model = SimCLR(proj_hidden_dim=2048, proj_out_dim=128)
        simclr_model.load_state_dict(torch.load(f"../experiment/simCLR/simclr_final.pth")["model"])
        
        # encoder部分のみを使用（projectorは除去）
        model = simclr_model.encoder
        
        # エンコーダー部分を凍結
        for param in model.parameters():
            param.requires_grad = False
            
        # 下流タスク用の分類ヘッドを追加（ResNet18の特徴量次元は512）
        model.fc = nn.Linear(512, 2)  # 2クラス分類（バイナリ）
        
    else:
        raise ValueError("model_type must be 'imagenet', 'scratch', 'SSL', 'radimagenet', or 'simCLR'")
    
    return model

def linear_probing(dataset, model_type, saved_epoch_num=None, epochs=10, batchsize=32):
    """Linear probingを実行する関数"""
    if dataset == "thyroid":
        dataset_class = dd.ThyroidDataset
    elif dataset == "breast":
        dataset_class = dd.BreastUSDataset
    elif dataset == "acl":
        dataset_class = dd.ACLDataset
    else:
        raise ValueError("dataset must be 'thyroid', 'breast', or 'acl'")
    
    # 5foldのtrain/val/testを行う
    preds_labels_all = []
    
    for i in range(5):
        print(f"Processing fold {i+1}/5...")
        
        # データセットの読み込み
        train_dataset = dataset_class(i+1, "train", transform=transform)
        val_dataset = dataset_class(i+1, "val", transform=transform)
        test_dataset = dataset_class(i+1, "test", transform=transform)
        
        # データローダーの作成
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
        
        # モデルを読み込み
        model = load_model(model_type, saved_epoch_num)
        model = model.cuda()
        cudnn.benchmark = True
        
        # 学習可能なパラメータのみを最適化対象に
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        
        # 損失関数の定義
        criterion = nn.CrossEntropyLoss()
        # 最適化手法の定義（学習率を高めに設定）
        optimizer = optim.Adam(trainable_params, lr=0.001)
        # 学習率のスケジューラーの定義
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # 学習ループ
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # 学習
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
            
            avg_train_loss = running_loss / len(train_dataloader)
            
            # 検証
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to("cuda:0")
                    labels = labels.to("cuda:0")
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # 最良モデルの保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
            
            scheduler.step()
        
        # 最良モデルでテスト
        model.load_state_dict(best_model_state)
        model.eval()
        
        test_loss = 0.0
        test_corrects = 0
        preds_fold = []
        labels_fold = []
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)
                preds_fold.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                labels_fold.extend(labels.cpu().numpy())
        
        test_loss = test_loss / len(test_dataset)
        test_acc = test_corrects.double() / len(test_dataset)
        print(f"Fold {i+1} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        preds_labels_all.append([preds_fold, labels_fold])
    
    return preds_labels_all

def main():
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
        print(f"--- Running Linear Probing Experiment {exp_num}/{args.num_experiments} ---")
        print(f"Dataset: {args.dataset}, Model: {args.model_type}")
        
        # Linear probingを実行
        result = linear_probing(
            dataset=args.dataset, 
            model_type=args.model_type, 
            saved_epoch_num=args.model_num, 
            epochs=args.epochs, 
            batchsize=args.batchsize
        )
        
        # 結果の保存
        model_path = f"../results/{args.dataset}/linear_probing_{args.model_type}_{args.model_num}/{exp_num}"
        os.makedirs(model_path + f"/{args.dataset}", exist_ok=True)
        print(f"Created directory: {model_path}/{args.dataset}")
        
        # foldごとの結果を保存
        for i in range(5):
            with open(f"{model_path}/{args.dataset}/fold_{i+1}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["preds", "labels"])
                for j in range(len(result[i][0])):
                    writer.writerow([result[i][0][j], result[i][1][j]])
        
        # 各foldでのacc, auc, sensitivity, specificityを計算
        accs = []
        aucs = []
        sensitivities = []
        specificities = []
        
        for i in range(5):
            preds = np.array(result[i][0])
            labels = np.array(result[i][1])
            
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
        
        print(f"Results - Acc: {acc_mean:.4f}, AUC: {auc_mean:.4f}, Sensitivity: {sensitivity_mean:.4f}, Specificity: {specificity_mean:.4f}")
        
        # 統計結果をcsvに保存
        with open(f"{model_path}/{args.dataset}/stats.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["acc", "auc", "sensitivity", "specificity"])
            for i in range(5):
                writer.writerow([accs[i], aucs[i], sensitivities[i], specificities[i]])
            writer.writerow([acc_mean, auc_mean, sensitivity_mean, specificity_mean])
        
        # ROC曲線を描画
        plt.figure()
        for i in range(5):
            preds = np.array(result[i][0])
            labels = np.array(result[i][1])
            fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
            auc = metrics.auc(fpr, tpr)
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
        plt.plot(fpr, tpr, label=f"mean (area = {auc:.2f})", linestyle="--", color="blue")
        
        plt.legend()
        plt.title(f"Linear Probing ROC Curve - {args.model_type.upper()}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.savefig(f"{model_path}/{args.dataset}/roc_curve.png")
        plt.close()
        
        print(f"Results saved to {model_path}/{args.dataset}/")

if __name__ == "__main__":
    main()
