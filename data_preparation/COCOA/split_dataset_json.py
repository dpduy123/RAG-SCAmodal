import os
import json
import argparse

def split_coco_json(input_json_path, output_dir, chunk_size=5000):
    print(f"Loading {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
        
    annotations = data.get('annotations', [])
    total_anns = len(annotations)
    
    if total_anns == 0:
        print("Không tìm thấy annotations nào để cắt.")
        return
        
    print(f"Total annotations: {total_anns}. Splitting into parts of {chunk_size}...")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_json_path).replace('.json', '')
    
    part_idx = 1
    for i in range(0, total_anns, chunk_size):
        chunk_anns = annotations[i:i+chunk_size]
        
        # Tạo object json mới cho từng part
        new_data = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": data.get("categories", []),
            "images": data.get("images", []), # Giữ nguyên danh sách images để map file_name
            "annotations": chunk_anns
        }
        
        output_file = os.path.join(output_dir, f"{base_name}_part{part_idx}.json")
        print(f"Saving part {part_idx} to {output_file} (from idx {i} to {i+len(chunk_anns)})...")
        
        with open(output_file, 'w') as out_f:
            json.dump(new_data, out_f)
            
        part_idx += 1
        
    print(f"✅ Đã chia thành {part_idx-1} parts thành công tại {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chia nhỏ file JSON COCOA/fullCOCOA")
    parser.add_argument("--input", type=str, default="dataset/fullCOCOA/annotations/COCO_amodal_test2014.json", help="Đường dẫn file cần chia")
    parser.add_argument("--output_dir", type=str, default="dataset/COCOA", help="Thư mục lưu các phần")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Số lượng annotations mỗi phần")
    args = parser.parse_args()
    
    split_coco_json(args.input, args.output_dir, args.chunk_size)
