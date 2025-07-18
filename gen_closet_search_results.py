import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64

# ==== 設定參數 ====
query_json = "clip_features.json"               # 查詢：分割後衣物（seg_pic）
db_json = "closet_clip_features.json"           # 資料庫：衣櫃原圖（closet）
query_img_dir = "seg_pic"
db_img_dir = "closet"
output_html = "clip_to_closet_results.html"
top_k = 3

# ==== 載入資料 ====
with open(query_json, "r") as f:
    query_data = json.load(f)
with open(db_json, "r") as f:
    db_data = json.load(f)

db_keys = list(db_data.keys())
db_vectors = np.array([db_data[k]["feature"] for k in db_keys])

# ==== 圖片轉 base64 工具 ====
def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img_f:
        return f"data:image/png;base64,{base64.b64encode(img_f.read()).decode()}"

# ==== HTML 開頭 ====
full_html = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <title>分割衣物 → Closet 相似搜尋結果</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        h2 { margin-top: 40px; }
        .result-block { margin-bottom: 40px; }
        .img-row { display: flex; gap: 20px; margin-top: 10px; }
        .img-col { text-align: center; }
        img { height: 200px; border: 1px solid #ccc; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>📦 分割衣物 → 衣櫃 Top-K 相似推薦</h1>
"""

# ==== 每筆查詢執行比對 ====
for query_key, query_info in query_data.items():
    query_vec = np.array(query_info["feature"]).reshape(1, -1)
    sims = cosine_similarity(query_vec, db_vectors)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    query_img_path = os.path.join(query_img_dir, f"{query_key}.png")
    query_img_base64 = img_to_base64(query_img_path)

    block = f"""
    <div class="result-block">
        <h2>查詢圖片：{query_key}.png</h2>
        <div class="img-row">
            <div class="img-col">
                <img src="{query_img_base64}">
                <div><strong>🔍 Segmented Item</strong></div>
            </div>
    """

    for rank, idx in enumerate(top_indices, 1):
        db_key = db_keys[idx]
        db_item = db_data[db_key]
        db_img_path = os.path.join(db_img_dir, db_item["filename"])
        db_img_base64 = img_to_base64(db_img_path)
        sim_score = sims[idx]

        block += f"""
            <div class="img-col">
                <img src="{db_img_base64}">
                <div><strong>Top {rank}</strong><br>相似度: {sim_score:.3f}</div>
            </div>
        """

    block += "</div></div>"
    full_html += block

# ==== HTML 結尾 ====
full_html += "</body></html>"

# ==== 儲存 HTML ====
with open(output_html, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"✅ 已產出搜尋結果頁：{output_html}")
print("📂 請用瀏覽器開啟檢視推薦結果")
