import os
import shutil
import glob

def organize_dataset(dataset_dir):
    # 移動先のディレクトリを作成
    avi_root = os.path.join(dataset_dir, 'avi')
    mp4_root = os.path.join(dataset_dir, 'mp4')
    
    os.makedirs(avi_root, exist_ok=True)
    os.makedirs(mp4_root, exist_ok=True)
    
    # datasetディレクトリ直下のアイテムを取得
    # avi, mp4 ディレクトリ自体や、ファイル（Annot_List.txtなど）は除外して探索する必要がある
    # ここでは os.walk を使って再帰的に探索するが、移動先の avi/mp4 フォルダに入らないように注意する
    
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        # 移動先ディレクトリ自体はスキップ
        if root.startswith(avi_root) or root.startswith(mp4_root):
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            # dataset直下のファイル（Annot_List.txtなど）は移動しない
            if root == dataset_dir:
                continue
                
            # 元のサブディレクトリ構造を維持するための相対パス
            # 例: root = .../dataset/videos 3 -> rel_path = videos 3
            rel_path = os.path.relpath(root, dataset_dir)
            
            if ext == '.avi':
                dest_dir = os.path.join(avi_root, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, file)
                
                print(f"Moving {file} to {dest_path}")
                shutil.move(file_path, dest_path)
                
            elif ext == '.mp4':
                dest_dir = os.path.join(mp4_root, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, file)
                
                print(f"Moving {file} to {dest_path}")
                shutil.move(file_path, dest_path)

        # ディレクトリが空になったか確認して削除
        # dataset直下のディレクトリのみ削除対象とする（再帰的に空なら消える）
        if root != dataset_dir:
            if not os.listdir(root):
                print(f"Removing empty directory: {root}")
                os.rmdir(root)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
    else:
        organize_dataset(dataset_dir)
