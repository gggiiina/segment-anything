import os
import torch
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

print("🚀 初始化 YOLOS 模型中...")
processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
print("✅ 模型載入完成")

# === 設定目錄 ===
input_dir = "test_image"
output_json_path = "sam_data_already_prepro_shoes.json"

# 你想要保留的類別 ID（僅服飾）
target_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 23]

'''
0: shirt, blouse
1: top, t-shirt, sweatshirt
2: sweater
3: cardigan
4: jacket
5: vest
6: pants
7: shorts
8: skirt
9: coat
10: dress
11: jumpsuit
12: cape
13: glasses
14: hat
15: headband, head covering, hair accessory
16: tie
17: glove
18: watch
19: belt
20: leg warmer
21: tights, stockings
22: sock
23: shoe
24: bag, wallet
25: scarf
26: umbrella
27: hood
28: collar
29: lapel
30: epaulette
31: sleeve
32: pocket
33: neckline
34: buckle
35: zipper
36: applique
37: bead
38: bow
39: flower
40: fringe
41: ribbon
42: rivet
43: ruffle
44: sequin
45: tassel
'''

# 載入字型（可選）
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# 儲存所有圖片的偵測資料
all_sam_data = []

# 取得所有圖片檔案
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"📂 偵測到 {len(image_files)} 張圖片要處理...\n")

# === 處理每張圖片 ===
for idx, filename in enumerate(image_files, start=1):
    print(f"🖼️ [{idx}/{len(image_files)}] 處理圖片：{filename}")
    image_path = os.path.join(input_dir, filename)
    image = Image.open(image_path).convert("RGB")

    # 處理與推論
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print("  🔍 推論完成")

    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    print(f"  🎯 篩選信心分數 > 0.5 的物件，共 {len(results['scores'])} 個")

    best_detections = {}
    shoe_candidates = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_id = int(label.item())
        if label_id not in target_ids:
            continue

        score_val = float(score.item())
        label_name = model.config.id2label[label_id]

        if label_id == 23:
            shoe_candidates.append((score_val, box))
        else:
            if label_id not in best_detections or score_val > best_detections[label_id][0]:
                best_detections[label_id] = (score_val, box, label_name)

    # 處理鞋子的前兩個 bbox
    shoe_candidates.sort(reverse=True, key=lambda x: x[0])
    if len(shoe_candidates) >= 2:
        # 取分數前兩高
        _, box1 = shoe_candidates[0]
        _, box2 = shoe_candidates[1]
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        merged_box = torch.tensor([x1, y1, x2, y2])
        best_detections[23] = (1.0, merged_box, "shoe")

    print(f"  ✅ 保留目標類別 {len(best_detections)} 個")

    draw = ImageDraw.Draw(image)

    for label_id, (score_val, box_tensor, label_name) in best_detections.items():
        box = [round(i, 2) for i in box_tensor.tolist()]
        text = f"{label_name} ({round(score_val, 2)})"

        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle(box, outline="red", width=3)
        draw.rectangle(
            [box[0], max(0, box[1] - text_height - 10), box[0] + text_width + 4, box[1]],
            fill="red"
        )
        draw.text((box[0] + 2, box[1] - text_height - 5), text, fill="white", font=font)

        all_sam_data.append({
            "filename": filename,
            "label": label_name,
            "box": box
        })

    # 若要儲存畫好框的圖可取消註解：
    # output_img_path = os.path.join("output_image", f"det_{filename}")
    # image.save(output_img_path)
    # print(f"  💾 已儲存帶框圖片：{output_img_path}")

    print(f"  📦 圖片 {filename} 處理完成\n")

# === 儲存 JSON ===
with open(output_json_path, "w") as f:
    json.dump(all_sam_data, f, indent=2)

print(f"🎉 全部圖片處理完成！")
print(f"📝 偵測資料已儲存為：{output_json_path}")