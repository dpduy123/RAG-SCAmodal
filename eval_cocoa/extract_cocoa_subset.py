import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import os
import re
from tqdm import tqdm

def extract_subset():
    list_file = "dataset/COCOA/img_filenames_cocoa.txt"
    full_ann_dir = "dataset/fullCOCOA/annotations"
    output_file = "dataset/COCOA/cocoa_annotation.json"
    
    if not os.path.exists(list_file):
        print(f"Error: {list_file} not found.")
        return

    # 1. Read required filenames
    with open(list_file, "r") as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Parse filenames into a searchable structure
    # Format: subset -> image_id -> list of names
    requests = {}
    for fname in filenames:
        clean_name = fname.replace("./", "")
        # Regex to extract: subset (train/val/test), image_id, and the suffix name
        match = re.search(r'COCO_(train|val|test)2014_(\d+)_?(.+)?\.jpg', clean_name)
        if match:
            subset = match.group(1)
            img_id = int(match.group(2))
            obj_name = match.group(3).replace("_", " ") if match.group(3) else None
            
            if subset not in requests: requests[subset] = {}
            if img_id not in requests[subset]: requests[subset][img_id] = []
            requests[subset][img_id].append({"full_name": clean_name, "obj_name": obj_name})

    # 3. Load full annotations and match
    final_annotations = []
    
    for subset in requests.keys():
        json_path = os.path.join(full_ann_dir, f"COCO_amodal_{subset}2014.json")
        if not os.path.exists(json_path):
            print(f"Warning: Annotation file {json_path} not found. Skipping {subset}.")
            continue
            
        print(f"Loading {json_path}...")
        with open(json_path, "r") as f:
            full_data = json.load(f)
            
        # COCOA format: list of segments directly or in 'annotations'
        raw_anns = full_data if isinstance(full_data, list) else full_data.get('annotations', [])
        
        # Create a lookup for speed
        subset_requests = requests[subset]
        
        print(f"Filtering {subset} annotations...")
        for ann in tqdm(raw_anns):
            ann_img_id = ann.get('image_id')
            if ann_img_id in subset_requests:
                # Potential match, check if name matches one of our requests for this image
                for req in subset_requests[ann_img_id]:
                    # Some versions use 'name', some use 'category_id'
                    ann_name = ann.get('name', "").lower()
                    req_name = str(req['obj_name']).lower() if req['obj_name'] else ""
                    
                    # Match if name is contained or if we don't have a name requirement
                    if not req_name or req_name in ann_name or ann_name in req_name:
                        # Copy annotation and add our filename for easier loading later
                        new_ann = ann.copy()
                        new_ann['filename'] = req['full_name']
                        final_annotations.append(new_ann)
                        # We found it, but there might be other requests for same image, so don't break yet

    # 4. Save result
    with open(output_file, "w") as f:
        json.dump({"annotations": final_annotations}, f, indent=4)
        
    print(f"\nSuccessfully extracted {len(final_annotations)} annotations to {output_file}")

if __name__ == "__main__":
    extract_subset()
