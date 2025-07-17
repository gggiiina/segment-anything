import json
import os
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

# === 載入整合資料 ===
with open("sam_data.json", "r") as f:
    sam_data = json.load(f)

print(f"✅ 共載入 {len(sam_data)} 個 box prompt")

# === 載入圖片 ===
image_path = os.path.join("test_image", "16.jpg")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ 圖片不存在：{image_path}")

image = cv2.imread(image_path)
if image is None:
    raise ValueError("❌ 圖片讀取失敗（可能是損壞或格式不支援）")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === 初始化 SAM ===
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# === 建立輸出資料夾 ===
output_dir = "seg_pic"
os.makedirs(output_dir, exist_ok=True)

# === 遮罩套用原圖、裁切並儲存 ===
label_count = {}

for i, item in enumerate(sam_data):
    label = item["label"]
    box = np.array(item["box"])
    box_input = box[None, :]  # (1, 4)

    masks, _, _ = predictor.predict(box=box_input, multimask_output=False)
    mask = masks[0]

    # 遮罩轉成 3-channel，與 RGB 原圖做遮罩相乘
    mask_3c = np.stack([mask] * 3, axis=-1)
    cutout = image_rgb * mask_3c

    # 根據框裁切
    x0, y0, x1, y1 = box.astype(int)
    cropped = cutout[y0:y1, x0:x1]

    # 避免同名 label 覆蓋
    count = label_count.get(label, 0)
    save_name = f"{label}_{count}.png"
    label_count[label] = count + 1

    # 儲存圖片
    save_path = os.path.join(output_dir, save_name)
    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, cropped_bgr)
    print(f"✅ 已儲存：{save_path}")
