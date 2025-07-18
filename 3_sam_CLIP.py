import os
import json
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from segment_anything import SamPredictor, sam_model_registry
import torch
import clip

def segment_and_save(
    sam_data_path="sam_data.json",
    image_dir="test_image",
    output_dir="seg_pic",
    sam_checkpoint="sam_vit_h_4b8939.pth"
):
    """使用 SAM 分割所有物件並儲存切圖到 seg_pic"""
    os.makedirs(output_dir, exist_ok=True)

    with open(sam_data_path, "r") as f:
        sam_data = json.load(f)

    grouped_data = defaultdict(list)
    for item in sam_data:
        grouped_data[item["filename"]].append(item)

    print(f"🧠 載入 SAM 模型中...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    print(f"✅ SAM 模型載入完成")

    global_idx = 0

    for image_idx, (filename, detections) in enumerate(grouped_data.items(), start=1):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ 圖片讀取失敗：{image_path}，跳過")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        for i, item in enumerate(detections):
            label = item["label"]
            box = np.array(item["box"])
            box_input = box[None, :]

            masks, _, _ = predictor.predict(box=box_input, multimask_output=False)
            mask = masks[0]

            mask_3c = np.stack([mask] * 3, axis=-1)
            cutout = image_rgb * mask_3c
            x0, y0, x1, y1 = box.astype(int)
            cropped = cutout[y0:y1, x0:x1]

            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                print("⚠️ 區塊太小，略過")
                continue

            clean_label = label.replace("/", "-").replace("\\", "-")
            out_name = f"{global_idx}_{clean_label}_0.png"
            out_path = os.path.join(output_dir, out_name)

            success = cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            if success:
                print(f"✅ 已儲存：{out_path}")
                global_idx += 1
            else:
                print(f"❌ 儲存失敗：{out_path}")


def extract_clip_features(seg_pic_dir="seg_pic", output_json="clip_features.json"):
    """針對 seg_pic 裡的所有圖像做 CLIP 特徵抽取，並儲存成 JSON"""
    print("🧠 載入 CLIP 模型中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ CLIP 模型載入完成（使用裝置：{device}）")

    features_dict = {}

    files = sorted([
        f for f in os.listdir(seg_pic_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    for idx, fname in enumerate(files, start=1):
        fpath = os.path.join(seg_pic_dir, fname)
        pil_img = Image.open(fpath).convert("RGB")

        image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image_tensor).cpu().numpy()[0]

        key = os.path.splitext(fname)[0]
        features_dict[key] = {
            "filename": fname,
            "feature": feature.tolist()
        }

        print(f"🔍 [{idx}/{len(files)}] 特徵已抽取：{key}")

    with open(output_json, "w") as f:
        json.dump(features_dict, f)

    print(f"\n🎉 全部特徵已抽取，儲存至：{output_json}")


# 🏁 範例執行流程（可放 main 函式中）
if __name__ == "__main__":
    segment_and_save()
    extract_clip_features()
