# ../results/の中にacl, breast, thyroidのフォルダがある
# それぞれのフォルダの中にimagenetモデル、scratchモデル、SSLモデルをファインチューニングした際の結果がある
# それぞれの結果は10回分格納されている
# 各試行でstats.csvが作成されており、最終行がacc,auc,sensitivity,specificityの平均値となっている。

# これらの結果をwilcoxon signed rank testで比較する
# まずはaclの結果を比較する

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_num", type=str, default="acl")
args = parser.parse_args()

# aclの10回分の結果を読み込む
acl_imagenet_list = []
acl_scratch_list = []
acl_ssl_list = []

# stats.csvの最終行を取得
for i in range(10):
    acl_imagenet_list.append(pd.read_csv(f"../results/acl/miniRIN_HML_imagenet/{i+1}/acl/stats.csv").iloc[-1, :])
    acl_scratch_list.append(pd.read_csv(f"../results/acl/miniRIN_HML_scratch/{i+1}/acl/stats.csv").iloc[-1, :])
    acl_ssl_list.append(pd.read_csv(f"../results/acl/miniRIN_HML_SSL_{args.model_num}/{i+1}/acl/stats.csv").iloc[-1, :])

# listをDataFrameに変換
acl_imagenet_df = pd.DataFrame(acl_imagenet_list)
acl_scratch_df = pd.DataFrame(acl_scratch_list)
acl_ssl_df = pd.DataFrame(acl_ssl_list)

# 各モデルのacc,auc,sensitivity,specificityの平均値を計算
acl_imagenet_mean = acl_imagenet_df.mean(axis=0)
acl_scratch_mean = acl_scratch_df.mean(axis=0)
acl_ssl_mean = acl_ssl_df.mean(axis=0)


# breastも同様に読み込む
breast_imagenet_list = []
breast_scratch_list = []
breast_ssl_list = []

# stats.csvの最終行を取得
for i in range(10):
    breast_imagenet_list.append(pd.read_csv(f"../results/breast/miniRIN_HML_imagenet/{i+1}/breast/stats.csv").iloc[-1, :])
    breast_scratch_list.append(pd.read_csv(f"../results/breast/miniRIN_HML_scratch/{i+1}/breast/stats.csv").iloc[-1, :]) 
    breast_ssl_list.append(pd.read_csv(f"../results/breast/miniRIN_HML_SSL/{i+1}/breast/stats.csv").iloc[-1, :])

# listをDataFrameに変換
breast_imagenet_df = pd.DataFrame(breast_imagenet_list)
breast_scratch_df = pd.DataFrame(breast_scratch_list)
breast_ssl_df = pd.DataFrame(breast_ssl_list)

# 各モデルのacc,auc,sensitivity,specificityの平均値を計算
breast_imagenet_mean = breast_imagenet_df.mean(axis=0)
breast_scratch_mean = breast_scratch_df.mean(axis=0)
breast_ssl_mean = breast_ssl_df.mean(axis=0)

# thyroidも同様に読み込む
thyroid_imagenet_list = []
thyroid_scratch_list = []
thyroid_ssl_list = []

# stats.csvの最終行を取得
for i in range(10):
    thyroid_imagenet_list.append(pd.read_csv(f"../results/thyroid/miniRIN_HML_imagenet/{i+1}/thyroid/stats.csv").iloc[-1, :])
    thyroid_scratch_list.append(pd.read_csv(f"../results/thyroid/miniRIN_HML_scratch/{i+1}/thyroid/stats.csv").iloc[-1, :])
    thyroid_ssl_list.append(pd.read_csv(f"../results/thyroid/miniRIN_HML_SSL/{i+1}/thyroid/stats.csv").iloc[-1, :])

# listをDataFrameに変換
thyroid_imagenet_df = pd.DataFrame(thyroid_imagenet_list)
thyroid_scratch_df = pd.DataFrame(thyroid_scratch_list)
thyroid_ssl_df = pd.DataFrame(thyroid_ssl_list)

# 各モデルのacc,auc,sensitivity,specificityの平均値を計算
thyroid_imagenet_mean = thyroid_imagenet_df.mean(axis=0)
thyroid_scratch_mean = thyroid_scratch_df.mean(axis=0)
thyroid_ssl_mean = thyroid_ssl_df.mean(axis=0)


# wilcoxon signed rank testを行う
# 検定の結果をtxtに保存
from scipy.stats import wilcoxon

# aclの結果を比較（SSLのaucが有意に高いかどうか）
# 片側
with open(f"acl_{args.model_num}.txt", "w") as f:
    f.write("acl\n")
    f.write(str(wilcoxon(acl_ssl_df["auc"], acl_imagenet_df["auc"], alternative="greater")))
    f.write(str(wilcoxon(acl_ssl_df["auc"], acl_scratch_df["auc"], alternative="greater")))

    f.write("breast\n")
    f.write(str(wilcoxon(breast_ssl_df["auc"], breast_imagenet_df["auc"], alternative="greater")))
    f.write(str(wilcoxon(breast_ssl_df["auc"], breast_scratch_df["auc"], alternative="greater")))

    f.write("thyroid\n")
    f.write(str(wilcoxon(thyroid_ssl_df["auc"], thyroid_imagenet_df["auc"], alternative="greater")))
    f.write(str(wilcoxon(thyroid_ssl_df["auc"], thyroid_scratch_df["auc"], alternative="greater")))