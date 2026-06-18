import os
import json
import glob
import re

def main():
    test_json_path = "dataset/COCOA_test_set/cocoa_annotation.json"
    target_dir = "dataset/COCOA"
    
    print(f"🔍 Bước 1: Quét danh sách test images từ {test_json_path}...")
    
    test_filenames = set()
    
    if os.path.exists(test_json_path):
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)
            
        for ann in test_data.get('annotations', []):
            url = ann.get('url', '')
            if url:
                # url: https://s3-us-west-1.amazonaws.com/coco-ann/coco-train/COCO_train2014_000000001145.jpg
                filename = url.split('/')[-1]
                test_filenames.add(filename)
                
    # Parse thêm từ txt file để chắc chắn 100%
    txt_path = "dataset/COCOA_test_set/img_filenames_cocoa.txt"
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                name = line.strip().split('/')[-1]
                # Extract base name from string like COCO_train2014_000000001145_whitecabinet.jpg
                # match r'(COCO_.*?\d+)\.jpg'
                # But it could also just be 'COCO_train2014_000000001145.jpg'
                match = re.search(r'(COCO_[a-zA-Z0-9_]*?\d{12})\.jpg', name)
                if match:
                    base_name = match.group(1) + ".jpg"
                    test_filenames.add(base_name)
                    
    print(f"✅ Đã tìm thấy {len(test_filenames)} hình ảnh test (cần loại bỏ để tránh Data Leakage).")
    
    print(f"\n🗑️ Bước 2: Bắt đầu lọc các file JSON trong {target_dir}...")
    json_files = glob.glob(os.path.join(target_dir, "*.json"))
    
    total_removed_imgs = 0
    total_removed_anns = 0
    
    for json_file in json_files:
        print(f"⏳ Processing {os.path.basename(json_file)}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        
        # 1. Tìm các image_id thuộc về test set
        test_image_ids = set()
        cleaned_images = []
        
        for img in images:
            file_name = img.get('file_name', '')
            if file_name in test_filenames:
                test_image_ids.add(img.get('id'))
            else:
                cleaned_images.append(img)
                
        # 2. Lọc bỏ các annotations trỏ tới test images
        cleaned_annotations = []
        for ann in annotations:
            if ann.get('image_id') not in test_image_ids:
                cleaned_annotations.append(ann)
                
        removed_imgs = len(images) - len(cleaned_images)
        removed_anns = len(annotations) - len(cleaned_annotations)
        
        if removed_imgs > 0 or removed_anns > 0:
            print(f"  ❌ Đã xoá {removed_imgs} images và {removed_anns} annotations.")
            data['images'] = cleaned_images
            data['annotations'] = cleaned_annotations
            
            # Ghi đè lại file
            with open(json_file, 'w') as f:
                json.dump(data, f)
                
            total_removed_imgs += removed_imgs
            total_removed_anns += removed_anns
        else:
            print(f"  ✅ Không có hình ảnh test nào trong file này.")
            
    print(f"\n🎉 HOÀN TẤT! Đã xoá tổng cộng {total_removed_imgs} images và {total_removed_anns} annotations từ tất cả các parts.")

if __name__ == "__main__":
    main()
