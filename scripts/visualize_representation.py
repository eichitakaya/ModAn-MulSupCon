# ../experiment/run_0/t_checkpoint.pth.tarのモデルを読み込む（tは1-10）
# ../../data/miniRINの画像に対するベクトルを取得
# UMAPで可視化
# 5種類のモデル（scratch, imagenet, SSL, radimagenet, simclr）を1つの図に並べて可視化

import os
import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms
from datasets import miniRINmultilabel
from utils import one_hot2label
import argparse
from train_simCLR import SimCLR
import matplotlib.patches as mpatches
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="all")
args = parser.parse_args()


transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),])


def load_model(model_name):
    """指定されたモデル名のモデルを読み込む"""
    if model_name == "radimagenet":
        model = resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 165)
        model.load_state_dict(torch.load(f"../experiment/radimagenet_sup_200epoch.pth"))
        model.fc = torch.nn.Identity()
        model.eval()

    elif model_name == "simclr":
        simclr_model = SimCLR(proj_hidden_dim=2048, proj_out_dim=128)
        simclr_model.load_state_dict(torch.load(f"../experiment/simCLR/simclr_final.pth")["model"])
        # encoder部分のみを使用（projectorは除去）
        model = simclr_model.encoder
        model.fc = torch.nn.Identity()
        model.eval()
        
    elif model_name == "imagenet":
        model = resnet18(pretrained=True)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Identity()
        model.eval()

    elif model_name == "scratch":
        model = resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Identity()
        model.eval()

    elif model_name == "SSL":
        model_path = f"/workspace/HMC4RadImageNet/experiment/run_0/200_lr005.tar"
        # resnet18を読み込む
        model = resnet18(weights=None)
        # 1チャンネルに変換
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
        model.fc = torch.nn.Identity()
        model.eval()
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return model


def get_embeddings(model, dataloader):
    """モデルを使用してデータセットから埋め込みベクトルを取得"""
    vectors = []
    targets = []
    
    for batch in dataloader:
        images = batch[0]
        target = batch[1]
        outputs = model(images)
        vectors.append(outputs[0].detach().cpu().numpy())
        target = one_hot2label(target)
        targets.append(target)
    
    return vectors, targets


def load_or_compute_embeddings(model_name, model, dataloader, cache_dir):
    """埋め込みベクトルをキャッシュから読み込むか、計算して保存する"""
    cache_file = os.path.join(cache_dir, f"{model_name}_embeddings.csv")
    
    # キャッシュファイルが存在する場合は読み込み
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings for {model_name}...")
        df = pd.read_csv(cache_file)
        vectors = [np.array(eval(vec)) for vec in df['vector']]
        targets = df['target'].tolist()
        return vectors, targets
    
    # キャッシュファイルが存在しない場合は計算
    print(f"Computing embeddings for {model_name}...")
    vectors, targets = get_embeddings(model, dataloader)
    
    # キャッシュに保存
    df = pd.DataFrame({
        'vector': [str(vec.tolist()) for vec in vectors],
        'target': targets
    })
    df.to_csv(cache_file, index=False)
    print(f"Saved embeddings to {cache_file}")
    
    return vectors, targets


def visualize_single_model(model_name, vectors, targets, ax, label_dict, cmap=None):
    """単一モデルの可視化"""
    # UMAPで可視化
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings = reducer.fit_transform(vectors)
    
    # 数値ラベルに変換
    numeric_targets = [label_dict[target] for target in targets]
    
    # プロット
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('tab20', 11)  # 11色で分割
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=numeric_targets, cmap=cmap, s=3, alpha=0.7)
    title_dic = {"SSL": "ModAn-MulSupCon", "simclr": "miniRINSimCLR", "radimagenet": "miniRIN165", "imagenet": "ImageNet", "scratch": "Scratch"}
    ax.set_title(f'{title_dic[model_name]}', fontsize=20, fontweight='bold')
    #ax.set_xlabel('UMAP1')
    #ax.set_ylabel('UMAP2')
    ax.grid(True, alpha=0.3)
    
    return scatter


def main():
    # データセットの読み込み
    dataset = miniRINmultilabel(transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 出力ディレクトリ
    out_dir = "../visualize"
    cache_dir = "../cache/embeddings"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # ラベル辞書を作成（全モデルで共通）
    # まずキャッシュから読み込みを試行
    label_cache_file = os.path.join(cache_dir, "label_dict.csv")
    if os.path.exists(label_cache_file):
        print("Loading cached label dictionary...")
        label_df = pd.read_csv(label_cache_file)
        label_dict = {row['target']: row['index'] for _, row in label_df.iterrows()}
    else:
        print("Computing label dictionary...")
        _, targets = get_embeddings(load_model("scratch"), dataloader)
        unique_targets = sorted(set(targets))
        label_dict = {target: i for i, target in enumerate(unique_targets)}
        
        # ラベル辞書をキャッシュに保存
        label_df = pd.DataFrame([
            {'target': target, 'index': idx} for target, idx in label_dict.items()
        ])
        label_df.to_csv(label_cache_file, index=False)
        print(f"Saved label dictionary to {label_cache_file}")
    
    print("Label dictionary:", label_dict)
    
    # モデル名のリスト（表示順）
    model_names = ["SSL", "simclr", "radimagenet", "imagenet", "scratch"]
    
    # 一貫したカラーマップと凡例用ハンドル
    cmap_shared = matplotlib.cm.get_cmap('tab20', 11)
    legend_handles = [
        mpatches.Patch(color=cmap_shared(label_dict[target]), label=str(target))
        for target in sorted(label_dict.keys())
    ]
    
    # 2x3のサブプロットを作成（上の段に3つ、下の段に2つ）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    #fig.suptitle('Representation Visualization Comparison', fontsize=16, fontweight='bold')
    
    # 各モデルを可視化
    for i, model_name in enumerate(model_names):
        print(f"Processing {model_name}...")
        
        # モデルを読み込み
        model = load_model(model_name)
        
        # 埋め込みベクトルを取得（キャッシュから読み込みまたは計算）
        vectors, targets = load_or_compute_embeddings(model_name, model, dataloader, cache_dir)
        
        # 位置を計算（上の段に3つ、下の段に2つ）
        if i < 3:
            # 上の段（0行目）
            row, col = 0, i
        else:
            # 下の段（1行目）
            row, col = 1, i - 3
        
        # 可視化（結合図の該当サブプロット）
        scatter = visualize_single_model(model_name, vectors, targets, axes[row, col], label_dict, cmap=cmap_shared)
        
        # 下の段の3つ目（右下）に凡例を表示（サイズを大きく）
        if i == 4:  # 最後のモデル（scratch）
            axes[row, col].legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)
        
        # 個別図を保存
        fig_single, ax_single = plt.subplots(figsize=(5, 5))
        scatter_single = visualize_single_model(model_name, vectors, targets, ax_single, label_dict, cmap=cmap_shared)
        # 5枚目（最後）のみ凡例を表示（サイズを大きく）
        if i == 4:
            ax_single.legend(handles=legend_handles, title='Class Labels', loc='best', fontsize=12)
        plt.tight_layout()
        single_path = os.path.join(out_dir, f"{model_name}.png")
        plt.savefig(single_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved {single_path}")
    
    # 下の段の3つ目（右下）を空にする（凡例用のスペース）
    axes[1, 2].set_visible(False)
    
    # レイアウトを調整
    plt.tight_layout()
    
    # 保存（結合図）
    combined_path = os.path.join(out_dir, "all_models_comparison.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved all_models_comparison.png")


if __name__ == "__main__":
    main()
