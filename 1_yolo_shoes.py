import os
import torch
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def detect_fashion_items(
    input_path,
    output_json_path="sam_data.json",
    output_image_dir="output_vis",
    conflict_rules_path="yolo_conflict_rules.json",
    show_image=False,
    save_image=False,
    min_confidence=0.5
):
    print("🚀 初始化 YOLOS 模型中...")
    processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
    print("✅ 模型載入完成")

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    target_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23]
    all_sam_data = []

    # ✅ 載入衝突規則
    if os.path.exists(conflict_rules_path):
        with open(conflict_rules_path, "r") as f:
            conflict_rules = json.load(f)
    else:
        conflict_rules = {}

    # ✅ 處理圖像來源
    if os.path.isdir(input_path):
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        raise ValueError(f"❌ 找不到指定路徑：{input_path}")
    print(f"📂 偵測到 {len(image_files)} 張圖片要處理...\n")

    if save_image and not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    for idx, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n🖼️ [{idx}/{len(image_files)}] 處理圖片：{filename}")
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]

        # === 第一步：收集所有候選項目（符合 label 和 min_confidence）
        all_candidates = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_id = int(label.item())
            score_val = float(score.item())

            if label_id not in target_ids:
                continue
            if score_val < min_confidence:
                continue

            label_name = model.config.id2label.get(label_id, f"id_{label_id}")
            all_candidates.append({
                "label_id": label_id,
                "score": score_val,
                "box": box,
                "label_name": label_name
            })

        # === 第二步：依 score 降冪排序
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        # === 第三步：依序檢查衝突規則來決定保留哪些項目
        best_detections = {}
        shoe_candidates = []

        for cand in all_candidates:
            label_id = cand["label_id"]
            score_val = cand["score"]
            box = cand["box"]
            label_name = cand["label_name"]

            print(f"🔍 嘗試加入：{label_name}（ID: {label_id}, 分數: {score_val:.2f}）")

            # 檢查是否已有相同的 label
            if label_id in best_detections:
                existing_score = best_detections[label_id][0]
                if score_val > existing_score:
                    print(f"  ⚠️ 已有相同標籤，但分數較高，取代原有項目")
                    best_detections[label_id] = (score_val, box, label_name)
                else:
                    print(f"  ⚠️ 已有相同標籤，分數較低，跳過此項目")
                continue

            # 檢查衝突
            conflict_with = None
            for existing_id in best_detections.keys():
                conflict_ids = conflict_rules.get(str(existing_id), [])
                if str(label_id) in conflict_ids:
                    conflict_with = existing_id
                    break

            if conflict_with is not None:
                # 比較分數，保留分數高的
                existing_score = best_detections[conflict_with][0]
                if score_val > existing_score:
                    print(f"  ⚠️ 與已選擇的 {conflict_with} 發生衝突，但分數較高，取代原有項目")
                    del best_detections[conflict_with]
                else:
                    print(f"  ⚠️ 與已選擇的 {conflict_with} 發生衝突，分數較低，跳過此項目")
                    continue

            # 處理鞋子特殊邏輯
            if label_id == 23:
                shoe_candidates.append((score_val, box))
                print(f"  ✅ 是鞋子，暫存起來備用")
            else:
                best_detections[label_id] = (score_val, box, label_name)
                print(f"  ✅ 加入 best_detections：{label_name}")


        # ✅ 合併兩隻鞋子
        shoe_candidates.sort(reverse=True, key=lambda x: x[0])
        if len(shoe_candidates) >= 2:
            _, box1 = shoe_candidates[0]
            _, box2 = shoe_candidates[1]
            x1 = min(box1[0], box2[0])
            y1 = min(box1[1], box2[1])
            x2 = max(box1[2], box2[2])
            y2 = max(box1[3], box2[3])
            merged_box = torch.tensor([x1, y1, x2, y2])
            best_detections[23] = (1.0, merged_box, "shoe")
            print(f"👟 合併鞋子成功，加入 best_detections")

        print("\n📌 本張圖的 best_detections 結果：")
        for k, (score_val, _, label_name) in best_detections.items():
            print(f"  - {label_name}（ID: {k}, 分數: {score_val:.2f}）")

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

        if save_image:
            save_path = os.path.join(output_image_dir, filename)
            image.save(save_path)
            print(f"💾 已儲存：{save_path}")

        if show_image:
            image.show()

    with open(output_json_path, "w") as f:
        json.dump(all_sam_data, f, indent=2)
    print(f"\n🎉 偵測完成，資料已儲存：{output_json_path}")

    return all_sam_data


# ✅ 執行範例
detect_fashion_items(
    input_path="test_image",
    output_json_path="sam_data.json",
    output_image_dir="output_yolo",
    conflict_rules_path="yolo_conflict_rules.json",
    show_image=False,
    save_image=True,
    min_confidence=0.05,
)
