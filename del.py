import os
import shutil

# 原始資料夾位置
src_base = r"C:\Users\user\Downloads\drive-download-20250718T055354Z-1-001"
# 目標儲存位置
dst_base = r"C:\Users\user\Desktop\segment-anything\closet"

# 類別資料夾名稱
categories = ["dress", "pants", "shoes", "skirt", "tops"]

# 建立 closet 主資料夾（如果不存在）
os.makedirs(dst_base, exist_ok=True)

for category in categories:
    src_folder = os.path.join(src_base, category)
    dst_folder = os.path.join(dst_base, category)
    os.makedirs(dst_folder, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(src_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for idx, filename in enumerate(image_files, start=1):
        # 編號補零到3位數
        new_filename = f"{category}_{idx:03d}.jpg"
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, new_filename)

        shutil.copy2(src_path, dst_path)  # 使用 copy2 保留 metadata
        print(f"✅ {src_path} → {dst_path}")
