import os
import torch
import clip
from PIL import Image
import json

# === 初始化 CLIP 模型 ===
print("🧠 載入 CLIP 模型中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"✅ CLIP 模型載入完成，使用裝置：{device}")

closet_dir = "closet"
output_json = "closet_clip_features.json"
closet_features = {}

# === 取得圖片列表 ===
image_files = [f for f in os.listdir(closet_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
print(f"🧾 共找到 {len(image_files)} 張圖片要處理")

# === 處理每張圖片 ===
for idx, fname in enumerate(image_files, start=1):
    image_path = os.path.join(closet_dir, fname)
    print(f"🔄 [{idx}/{len(image_files)}] 處理圖片：{fname}")

    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model.encode_image(image).cpu().numpy()[0]

        key = os.path.splitext(fname)[0]
        closet_features[key] = {
            "filename": fname,
            "feature": feature.tolist()
        }

        print(f"   ✅ 向量已抽取，儲存 key：{key}")

    except Exception as e:
        print(f"   ❌ 發生錯誤：{e}，略過此圖")

# === 儲存 JSON ===
with open(output_json, "w") as f:
    json.dump(closet_features, f)

print(f"\n🎉 全部處理完畢！共儲存 {len(closet_features)} 筆向量")
print(f"📁 特徵檔案已儲存至：{output_json}")
