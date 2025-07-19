from yolo.detect_fashion_items import detect_fashion_items
from sam.segment_objects import segment_objects

# 定義處理類型
process_type = "w_preprocess"  # 可以改成 "w_preprocess"

# 執行 YOLO 物件偵測
detect_fashion_items(
    input_path=f"become_image/{process_type}",
    output_json_path=f"sam_data_{process_type}.json",
    conflict_rules_path="yolo/yolo_conflict_rules.json",
    show_image=False,
    save_image=True,
    output_image_dir=f"output_image/yolo/{process_type}",
    min_confidence=0.05,
)

# 執行 SAM 分割
segment_objects(
    input_json_path=f"sam_data_{process_type}.json",
    input_image_dir=f"become_image/{process_type}",
    output_base_dir=f"output_image/sam/{process_type}",
    sam_checkpoint="sam_vit_h_4b8939.pth"
)