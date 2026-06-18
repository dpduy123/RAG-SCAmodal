import json
import glob
import os

def main():
    target_dir = "dataset/COCOA"
    json_files = glob.glob(os.path.join(target_dir, "*.json"))
    output_txt = os.path.join(target_dir, "img_filenames_cocoa.txt")
    
    unique_images = set()
    
    print(f"Bắt đầu quét qua {len(json_files)} file JSON trong thư mục {target_dir}...")
    
    for json_file in json_files:
        print(f"Đang đọc {os.path.basename(json_file)}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Lấy danh sách ảnh
        for img in data.get('images', []):
            filename = img.get('file_name')
            if filename:
                unique_images.add(filename)
                
    print(f"\n✅ Đã trích xuất thành công {len(unique_images)} tên file ảnh độc nhất!")
    
    with open(output_txt, 'w') as f:
        for img_name in sorted(unique_images):
            f.write(img_name + "\n")
            
    print(f"📁 Đã lưu danh sách vào: {output_txt}")

if __name__ == "__main__":
    main()
