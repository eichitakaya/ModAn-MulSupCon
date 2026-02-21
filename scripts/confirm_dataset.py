# /workspace/HMC4RadImageNet/downstrean_data/thyroid/dataframe/thyroid_test_fold1.csvと
# /workspace/HMC4RadImageNet/downstrean_data/thyroid/dataframe/thyroid_train_fold1.csvを結合して、行数を確認する。
# csvにはpath,benign or malignantが記載されているので、benignとmalignantの数を確認する。


import pandas as pd
# trainとtestについて、fold1から5までをそれぞれ結合する
df_train_concat = pd.DataFrame()
df_test_concat = pd.DataFrame()
# train
for i in range(1, 6):
    df_train = pd.read_csv("/workspace/HMC4RadImageNet/downstream_data/acl/dataframe/train_fold" + str(i) + ".csv")
    df_train_concat = pd.concat([df_train_concat, df_train])
print(f"trainの結合後の行数: {df_train_concat.shape}")
# 重複を削除
df_train_concat = df_train_concat.drop_duplicates()
print(f"trainの結合後の行数(重複削除後): {df_train_concat.shape}")

# labelの分布を確認する
print(df_train_concat["acl_label"].value_counts())