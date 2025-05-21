# RadImageNetを読み込み、各クラスの画像を100枚ずつ抽出して保存する
# RadImageNetと同様のフォルダ構造をコピーしておく

import os
import shutil
import random

# RadImageNetのパス
rad_path = "../../data/RadImageNet"

# 保存先のパス
save_path = "../../data/miniRIN"

# 保存先のフォルダを作成
os.makedirs(save_path, exist_ok=True)

# すべてのサブディレクトリを再帰的に探索
for root, dirs, files in os.walk(rad_path):
    # 画像ファイルがあるディレクトリのみ処理
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dcm'))]
    if len(image_files) == 0:
        continue

    # サンプリング（100枚未満なら全てコピー）
    sampled_files = random.sample(image_files, min(100, len(image_files)))

    # miniRIN側の保存先ディレクトリを作成
    rel_dir = os.path.relpath(root, rad_path)
    target_dir = os.path.join(save_path, rel_dir)
    os.makedirs(target_dir, exist_ok=True)

    # 画像をコピー
    for fname in sampled_files:
        src = os.path.join(root, fname)
        dst = os.path.join(target_dir, fname)
        shutil.copy2(src, dst)

#    
