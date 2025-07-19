import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# === 參數設定 ===
db_json = "closet_clip_features.json"         # 查詢資料庫變成 closet
query_json = "clip_features.json"             # 現在反過來：clip_features 是查詢端
db_img_dir = "closet"                          # 圖片路徑反過來
query_img_dir = "seg_pic"
top_k = 3

# === 載入資料 ===
with open(db_json, "r") as f:
    db_data = json.load(f)
with open(query_json, "r") as f:
    query_data = json.load(f)

db_keys = list(db_data.keys())
db_vectors = np.array([db_data[k]["feature"] for k in db_keys])

# === 對每一筆 query 做比對 ===
for query_key, query_info in query_data.items():
    print(f"\n🔍 查詢圖片：{query_info['filename']}")

    query_vec = np.array(query_info["feature"]).reshape(1, -1)
    similarities = cosine_similarity(query_vec, db_vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    for rank, idx in enumerate(top_indices, 1):
        db_key = db_keys[idx]
        db_item = db_data[db_key]
        db_img_path = os.path.join(db_img_dir, db_item["filename"])
        sim_score = similarities[idx]

        print(f"Top {rank}: {db_item['filename']} | 相似度: {sim_score:.3f}")
