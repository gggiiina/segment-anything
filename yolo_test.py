from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def visualize_yolos_detections(image_path, target_label=None, min_confidence=0.5, show=True):
    # 讀取圖片
    image = Image.open(image_path).convert("RGB")

    # 載入模型
    processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")

    # 推論
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # 後處理
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]

    # 可視化
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        score_val = float(score.item())
        if score_val < min_confidence:
            continue

        label_id = int(label.item())
        label_name = model.config.id2label[label_id]

        if target_label and label_name != target_label:
            continue

        text = f"{label_name} ({score_val:.2f})"
        box = [round(i, 2) for i in box.tolist()]
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle(box, outline="red", width=3)
        draw.rectangle(
            [box[0], max(0, box[1] - text_height - 10), box[0] + text_width + 4, box[1]],
            fill="red"
        )
        draw.text((box[0] + 2, box[1] - text_height - 5), text, fill="white", font=font)

        print(f"✔️ {label_name}: {score_val:.4f}")

    if show:
        image.show()

visualize_yolos_detections("test_image/24.png", target_label="top, t-shirt, sweatshirt", min_confidence=0.00, show=True)

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