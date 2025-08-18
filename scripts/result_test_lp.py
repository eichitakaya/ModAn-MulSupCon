# ../results/の中にacl, breast, thyroidのフォルダがある
# それぞれのフォルダの中にimagenetモデル、scratchモデル、SSLモデルをファインチューニングした際の結果がある
# それぞれの結果は10回分格納されている
# 各試行でstats.csvが作成されており、最終行がacc,auc,sensitivity,specificityの平均値となっている。

# これらの結果をwilcoxon signed rank testで比較する
# まずはaclの結果を比較する

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_num", type=str, default="200")
args = parser.parse_args()

# aclの10回分の結果を読み込む
acl_imagenet_list = []
acl_scratch_list = []
acl_ssl_list = []
acl_radimagenet_list = []
acl_simclr_list = []

# stats.csvの最終行を取得
for i in range(10):
    acl_imagenet_list.append(pd.read_csv(f"../results/acl/linear_probing_imagenet_1/{i+1}/acl/stats.csv").iloc[-1, :])
    acl_scratch_list.append(pd.read_csv(f"../results/acl/linear_probing_scratch_1/{i+1}/acl/stats.csv").iloc[-1, :])
    acl_ssl_list.append(pd.read_csv(f"../results/acl/linear_probing_SSL_{args.model_num}_lr005/{i+1}/acl/stats.csv").iloc[-1, :])
    acl_radimagenet_list.append(pd.read_csv(f"../results/acl/linear_probing_radimagenet_1/{i+1}/acl/stats.csv").iloc[-1, :])
    acl_simclr_list.append(pd.read_csv(f"../results/acl/linear_probing_simCLR_1_lr005/{i+1}/acl/stats.csv").iloc[-1, :]) 

# listをDataFrameに変換
acl_imagenet_df = pd.DataFrame(acl_imagenet_list)
acl_scratch_df = pd.DataFrame(acl_scratch_list)
acl_ssl_df = pd.DataFrame(acl_ssl_list)
acl_radimagenet_df = pd.DataFrame(acl_radimagenet_list)
acl_simclr_df = pd.DataFrame(acl_simclr_list)

# 各モデルのacc,auc,sensitivity,specificityの平均値を計算
acl_imagenet_mean = acl_imagenet_df.mean(axis=0)
acl_scratch_mean = acl_scratch_df.mean(axis=0)
acl_ssl_mean = acl_ssl_df.mean(axis=0)
acl_radimagenet_mean = acl_radimagenet_df.mean(axis=0)
acl_simclr_mean = acl_simclr_df.mean(axis=0)

# breastも同様に読み込む
breast_imagenet_list = []
breast_scratch_list = []
breast_ssl_list = []
breast_radimagenet_list = []
breast_simclr_list = []

# stats.csvの最終行を取得
for i in range(10):
    breast_imagenet_list.append(pd.read_csv(f"../results/breast/linear_probing_imagenet_1/{i+1}/breast/stats.csv").iloc[-1, :])
    breast_scratch_list.append(pd.read_csv(f"../results/breast/linear_probing_scratch_1/{i+1}/breast/stats.csv").iloc[-1, :]) 
    breast_ssl_list.append(pd.read_csv(f"../results/breast/linear_probing_SSL_{args.model_num}_lr005/{i+1}/breast/stats.csv").iloc[-1, :])
    breast_radimagenet_list.append(pd.read_csv(f"../results/breast/linear_probing_radimagenet_1/{i+1}/breast/stats.csv").iloc[-1, :])
    breast_simclr_list.append(pd.read_csv(f"../results/breast/linear_probing_simCLR_1_lr005/{i+1}/breast/stats.csv").iloc[-1, :])

# listをDataFrameに変換
breast_imagenet_df = pd.DataFrame(breast_imagenet_list)
breast_scratch_df = pd.DataFrame(breast_scratch_list)
breast_ssl_df = pd.DataFrame(breast_ssl_list)
breast_radimagenet_df = pd.DataFrame(breast_radimagenet_list)
breast_simclr_df = pd.DataFrame(breast_simclr_list)

# 各モデルのacc,auc,sensitivity,specificityの平均値を計算
breast_imagenet_mean = breast_imagenet_df.mean(axis=0)
breast_scratch_mean = breast_scratch_df.mean(axis=0)
breast_ssl_mean = breast_ssl_df.mean(axis=0)
breast_radimagenet_mean = breast_radimagenet_df.mean(axis=0)
breast_simclr_mean = breast_simclr_df.mean(axis=0)

# thyroidも同様に読み込む
thyroid_imagenet_list = []
thyroid_scratch_list = []
thyroid_ssl_list = []
thyroid_radimagenet_list = []
thyroid_simclr_list = []

# stats.csvの最終行を取得
for i in range(10):
    thyroid_imagenet_list.append(pd.read_csv(f"../results/thyroid/linear_probing_imagenet_1/{i+1}/thyroid/stats.csv").iloc[-1, :])
    thyroid_scratch_list.append(pd.read_csv(f"../results/thyroid/linear_probing_scratch_1/{i+1}/thyroid/stats.csv").iloc[-1, :])
    thyroid_ssl_list.append(pd.read_csv(f"../results/thyroid/linear_probing_SSL_{args.model_num}_lr005/{i+1}/thyroid/stats.csv").iloc[-1, :])
    thyroid_radimagenet_list.append(pd.read_csv(f"../results/thyroid/linear_probing_radimagenet_1/{i+1}/thyroid/stats.csv").iloc[-1, :])
    thyroid_simclr_list.append(pd.read_csv(f"../results/thyroid/linear_probing_simCLR_1_lr005/{i+1}/thyroid/stats.csv").iloc[-1, :])

# listをDataFrameに変換
thyroid_imagenet_df = pd.DataFrame(thyroid_imagenet_list)
thyroid_scratch_df = pd.DataFrame(thyroid_scratch_list)
thyroid_ssl_df = pd.DataFrame(thyroid_ssl_list)
thyroid_radimagenet_df = pd.DataFrame(thyroid_radimagenet_list)
thyroid_simclr_df = pd.DataFrame(thyroid_simclr_list)

# 各モデルのacc,auc,sensitivity,specificityの平均値を計算
thyroid_imagenet_mean = thyroid_imagenet_df.mean(axis=0)
thyroid_scratch_mean = thyroid_scratch_df.mean(axis=0)
thyroid_ssl_mean = thyroid_ssl_df.mean(axis=0)
thyroid_radimagenet_mean = thyroid_radimagenet_df.mean(axis=0)
thyroid_simclr_mean = thyroid_simclr_df.mean(axis=0)

# wilcoxon signed rank testを行う
# 検定の結果をtxtに保存
from scipy.stats import wilcoxon

# aclの結果を比較（SSLのaucが有意に高いかどうか）
# 片側
with open(f"test_linear_probing_{args.model_num}.txt", "w") as f:
    f.write("acl\n")
    f.write(f"SSL AUC mean: {acl_ssl_mean['auc']:.4f}\n")
    f.write(f"ImageNet AUC mean: {acl_imagenet_mean['auc']:.4f}\n")
    f.write(f"Scratch AUC mean: {acl_scratch_mean['auc']:.4f}\n")
    f.write(f"RadImageNet AUC mean: {acl_radimagenet_mean['auc']:.4f}\n")
    f.write(f"SimCLR AUC mean: {acl_simclr_mean['auc']:.4f}\n")
    f.write("\n")
    f.write("SSL vs imagenet: " + str(wilcoxon(acl_ssl_df["auc"], acl_imagenet_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs scratch: " + str(wilcoxon(acl_ssl_df["auc"], acl_scratch_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs radimagenet: " + str(wilcoxon(acl_ssl_df["auc"], acl_radimagenet_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs simclr: " + str(wilcoxon(acl_ssl_df["auc"], acl_simclr_df["auc"], alternative="greater")) + "\n")
    f.write("\n")
    f.write("breast\n")
    f.write(f"SSL AUC mean: {breast_ssl_mean['auc']:.4f}\n")
    f.write(f"ImageNet AUC mean: {breast_imagenet_mean['auc']:.4f}\n")
    f.write(f"Scratch AUC mean: {breast_scratch_mean['auc']:.4f}\n")
    f.write(f"RadImageNet AUC mean: {breast_radimagenet_mean['auc']:.4f}\n")
    f.write(f"SimCLR AUC mean: {breast_simclr_mean['auc']:.4f}\n")
    f.write("\n")
    f.write("SSL vs imagenet: " + str(wilcoxon(breast_ssl_df["auc"], breast_imagenet_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs scratch: " + str(wilcoxon(breast_ssl_df["auc"], breast_scratch_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs radimagenet: " + str(wilcoxon(breast_ssl_df["auc"], breast_radimagenet_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs simclr: " + str(wilcoxon(breast_ssl_df["auc"], breast_simclr_df["auc"], alternative="greater")) + "\n")
    f.write("\n")
    f.write("thyroid\n")
    f.write(f"SSL AUC mean: {thyroid_ssl_mean['auc']:.4f}\n")
    f.write(f"ImageNet AUC mean: {thyroid_imagenet_mean['auc']:.4f}\n")
    f.write(f"Scratch AUC mean: {thyroid_scratch_mean['auc']:.4f}\n")
    f.write(f"RadImageNet AUC mean: {thyroid_radimagenet_mean['auc']:.4f}\n")
    f.write(f"SimCLR AUC mean: {thyroid_simclr_mean['auc']:.4f}\n")
    f.write("\n")
    f.write("SSL vs imagenet: " + str(wilcoxon(thyroid_ssl_df["auc"], thyroid_imagenet_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs scratch: " + str(wilcoxon(thyroid_ssl_df["auc"], thyroid_scratch_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs radimagenet: " + str(wilcoxon(thyroid_ssl_df["auc"], thyroid_radimagenet_df["auc"], alternative="greater")) + "\n")
    f.write("SSL vs simclr: " + str(wilcoxon(thyroid_ssl_df["auc"], thyroid_simclr_df["auc"], alternative="greater")) + "\n")