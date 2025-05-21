from scipy.io import arff
import pandas as pd

# ARFF ファイルを読み込む
data, meta = arff.loadarff('../datasets/yeast/yeast-train.arff')

# NumPy recarray → DataFrame に変換
df = pd.DataFrame(data)

# バイト文字列（bytes）を文字列（str）に変換したい場合は以下を使う
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

# CSV として保存
df.to_csv('yeast-train.csv', index=False)