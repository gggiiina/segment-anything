import json
import os
import numpy as np
import cv2
from collections import defaultdict
from segment_anything import SamPredictor, sam_model_registry

# === 載入整合資料 ===
with open("sam_data_already_prepro_shoes.json", "r") as f:
    sam_data = json.load(f)

print(f"✅ 共載入 {len(sam_data)} 個 box prompt")

# === 初始化 SAM 模型 ===
print("🧠 載入 SAM 模型中...")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
print("✅ SAM 模型初始化完成")

# === 建立主輸出資料夾 ===
main_output_dir = "seg_pic2_shoes"
os.makedirs(main_output_dir, exist_ok=True)
print(f"📂 主輸出資料夾已準備好：{main_output_dir}")

# === sam_data 按圖片分組 ===
grouped_data = defaultdict(list)
for item in sam_data:
    grouped_data[item["filename"]].append(item)

print(f"📸 共偵測到 {len(grouped_data)} 張圖片需要處理\n")

# === 開始處理每張圖片 ===
for image_idx, (filename, detections) in enumerate(grouped_data.items(), start=1):
    print(f"🔄 [{image_idx}/{len(grouped_data)}] 處理圖片：{filename}，共 {len(detections)} 個物件")

    image_path = os.path.join("test_image", filename)
    if not os.path.exists(image_path):
        print(f"❌ 找不到圖片：{image_path}，跳過")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 圖片讀取失敗：{image_path}，跳過")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    print("   🧠 已設定圖片給 SAM Predictor")

    # 建立子資料夾：seg_16, seg_27, ...
    base_name = os.path.splitext(filename)[0]
    sub_output_dir = os.path.join(main_output_dir, f"seg_{base_name}")
    os.makedirs(sub_output_dir, exist_ok=True)
    print(f"   📁 建立子資料夾：{sub_output_dir}")

    # 儲存原圖到子資料夾
    original_img_output_path = os.path.join(sub_output_dir, filename)
    cv2.imwrite(original_img_output_path, image)
    print(f"   🖼️ 已儲存原圖：{original_img_output_path}")

    # 同 label 覆蓋策略：僅存第一個 label 對應圖片，命名為 label.png
    used_labels = set()

    for i, item in enumerate(detections):
        label = item["label"]
        if label in used_labels:
            print(f"      ⚠️ label {label} 已處理過，跳過")
            continue
        used_labels.add(label)

        box = np.array(item["box"])
        box_input = box[None, :]  # (1, 4)

        print(f"   🎯 [{i+1}/{len(detections)}] 處理 label: {label}，box: {box.tolist()}")

        # SAM 預測遮罩
        masks, _, _ = predictor.predict(box=box_input, multimask_output=False)
        mask = masks[0]

        # 建立遮罩區塊
        mask_3c = np.stack([mask] * 3, axis=-1)
        cutout = image_rgb * mask_3c

        # 裁切區域
        x0, y0, x1, y1 = box.astype(int)
        cropped = cutout[y0:y1, x0:x1]

        # 儲存圖片
        save_path = os.path.join(sub_output_dir, f"{label}.png")
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(save_path, cropped_bgr)

        if success:
            print(f"      ✅ 已儲存：{save_path}")
        else:
            print(f"      ❌ 儲存失敗：{save_path}")

    print(f"   ✅ 圖片 {filename} 處理完成\n")

print("🎉 全部圖片處理完畢！所有切割圖與原圖已輸出到：", main_output_dir)
