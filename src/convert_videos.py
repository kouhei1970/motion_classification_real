import os
import subprocess
import glob

def convert_videos(dataset_dir):
    # datasetディレクトリ以下のすべてのサブディレクトリを探索
    # globを使って再帰的に.aviファイルを探す
    avi_files = glob.glob(os.path.join(dataset_dir, '**', '*.avi'), recursive=True)
    
    print(f"Found {len(avi_files)} .avi files.")
    
    for avi_path in avi_files:
        mp4_path = os.path.splitext(avi_path)[0] + '.mp4'
        
        # すでにmp4が存在する場合はスキップ（必要に応じて上書き設定に変更可能）
        if os.path.exists(mp4_path):
            print(f"Skipping {avi_path} (mp4 already exists)")
            continue
            
        print(f"Converting {avi_path} to {mp4_path}...")
        
        try:
            # ffmpegコマンドを実行
            # -y: 上書き許可, -i: 入力ファイル
            # 出力ログを抑制するために -loglevel error を使用
            cmd = ['ffmpeg', '-y', '-i', avi_path, '-c:v', 'libx264', '-c:a', 'aac', '-loglevel', 'error', mp4_path]
            subprocess.run(cmd, check=True)
            print(f"Successfully converted: {os.path.basename(avi_path)}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {avi_path}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # スクリプトの場所からdatasetディレクトリへのパスを解決
    # src/convert_videos.py から見て ../dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
    else:
        convert_videos(dataset_dir)
