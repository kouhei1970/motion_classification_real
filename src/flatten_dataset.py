import os
import shutil

def flatten_directory(target_dir):
    print(f"Flattening {target_dir}...")
    
    files_to_move = []
    
    # まず移動対象のファイルをリストアップ
    for root, dirs, files in os.walk(target_dir):
        if root == target_dir:
            continue
            
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)
            files_to_move.append((src_path, dst_path))
    
    # ファイル移動
    for src, dst in files_to_move:
        if os.path.exists(dst):
            print(f"Warning: {dst} already exists. Skipping {src}.")
            continue
            
        # print(f"Moving {os.path.basename(src)} to {os.path.basename(target_dir)}/")
        shutil.move(src, dst)
        
    # 空ディレクトリ削除 (深い階層から順に削除するために topdown=False)
    for root, dirs, files in os.walk(target_dir, topdown=False):
        if root == target_dir:
            continue
            
        if not os.listdir(root):
            print(f"Removing empty directory: {root}")
            os.rmdir(root)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    avi_dir = os.path.join(dataset_dir, 'avi')
    mp4_dir = os.path.join(dataset_dir, 'mp4')
    
    if os.path.exists(avi_dir):
        flatten_directory(avi_dir)
    else:
        print(f"{avi_dir} does not exist.")
    
    if os.path.exists(mp4_dir):
        flatten_directory(mp4_dir)
    else:
        print(f"{mp4_dir} does not exist.")
