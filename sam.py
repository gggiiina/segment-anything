import json
import os
import numpy as np
import cv2
from collections import defaultdict
from segment_anything import SamPredictor, sam_model_registry

# === 載入整合資料 ===
with open("sam_data.json", "r") as f:
    sam_data = json.load(f)

print(f"✅ 共載入 {len(sam_data)} 個 box prompt")

# === 初始化 SAM 模型 ===
print("🧠 載入 SAM 模型中...")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
print("✅ SAM 模型初始化完成")

# === 建立輸出資料夾 ===
output_dir = "seg_pic"
os.makedirs(output_dir, exist_ok=True)
print(f"📂 輸出資料夾已準備好：{output_dir}")

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

    label_count = {}

    for i, item in enumerate(detections):
        label = item["label"]
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

        # 避免同 label 覆蓋
        base_name = os.path.splitext(filename)[0]
        count = label_count.get(label, 0)
        save_name = f"{base_name}_{label}_{count}.png"
        label_count[label] = count + 1

        # 儲存圖片
        save_path = os.path.join(output_dir, save_name)
        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(save_path, cropped_bgr)

        if success:
            print(f"      ✅ 已儲存：{save_path}")
        else:
            print(f"      ❌ 儲存失敗：{save_path}")

    print(f"   🖼️ 圖片 {filename} 處理完成\n")

print("🎉 全部圖片處理完畢！所有切割圖已輸出到：", output_dir)
