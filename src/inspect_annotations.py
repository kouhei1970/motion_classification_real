import os
import csv

def inspect_annotations(dataset_dir):
    annotation_files = ['Annot_List.txt']
    
    class_mapping = {}
    
    for filename in annotation_files:
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found.")
            continue
            
        print(f"Reading {filename}...")
        with open(filepath, 'r') as f:
            # CSVとして読み込む
            reader = csv.DictReader(f)
            for row in reader:
                # フォーマット: video,label,id,t_start,t_end,frames
                class_name = row['label'].strip()
                class_id = row['id'].strip()
                
                if class_id not in class_mapping:
                    class_mapping[class_id] = class_name
                else:
                    if class_mapping[class_id] != class_name:
                        print(f"Conflict found for ID {class_id}: {class_mapping[class_id]} vs {class_name}")

    print("\n--- Class Mapping (ID: Name) ---")
    # IDでソートして表示（数値としてソート）
    try:
        sorted_ids = sorted(class_mapping.keys(), key=lambda x: int(x))
    except ValueError:
        sorted_ids = sorted(class_mapping.keys())
        
    for cid in sorted_ids:
        print(f"ID {cid}: {class_mapping[cid]}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
    else:
        inspect_annotations(dataset_dir)
