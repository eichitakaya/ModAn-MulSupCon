import csv
import time
import pandas as pd

CSV_PATH = '/root/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/list_attr_celeba.csv'

def test_csv_reader():
    start_time = time.time()
    rows = 0
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        for row in reader:  # メモリ効率を考慮してイテレータとして使用
            rows += 1
    end_time = time.time()
    return end_time - start_time, rows

def test_pandas_read():
    start_time = time.time()
    df = pd.read_csv(CSV_PATH)
    end_time = time.time()
    return end_time - start_time, len(df)

print("ファイルの読み込み速度を比較します...")

# 各方法で3回ずつテスト
for i in range(3):
    print(f"\nTest {i+1}:")
    
    # csv.reader
    csv_time, csv_rows = test_csv_reader()
    print(f"csv.reader: {csv_time:.4f}秒 ({csv_rows}行)")
    
    # pandas
    pd_time, pd_rows = test_pandas_read()
    print(f"pandas: {pd_time:.4f}秒 ({pd_rows}行)")
    
    if pd_time < csv_time:
        print(f"速度比: pandas は csv.reader の {csv_time/pd_time:.2f}倍速い")
    else:
        print(f"速度比: csv.reader は pandas の {pd_time/csv_time:.2f}倍速い") 