import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import json
import glob

# 設定
TARGET_FRAMES = 100
OUTPUT_DIR = 'data'
DATASET_DIR = 'dataset'
# アノテーションがAVIファイルのフレーム番号に基づいているため、AVIを使用する
VIDEO_DIR = os.path.join(DATASET_DIR, 'avi')
VIDEO_EXT = '.avi'
ANNOT_FILE = os.path.join(DATASET_DIR, 'Annot_List.txt')

def load_annotations():
    """
    Annot_List.txtを読み込み、動画ごとのアノテーションリストと
    クラスIDのマッピングを作成する。
    D0Xは除外する。
    """
    if not os.path.exists(ANNOT_FILE):
        print(f"Annotation file not found: {ANNOT_FILE}")
        return {}, {}

    video_annotations = {}
    unique_labels = set()

    print(f"Loading annotations from {ANNOT_FILE}...")
    with open(ANNOT_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['video']
            label = row['label']
            
            # D0Xは除外
            if label == 'D0X':
                continue
                
            unique_labels.add(label)
            
            try:
                start_frame = int(row['t_start'])
                end_frame = int(row['t_end'])
            except ValueError:
                continue
                
            if video_name not in video_annotations:
                video_annotations[video_name] = []
            
            video_annotations[video_name].append({
                'label': label,
                'start': start_frame,
                'end': end_frame
            })
            
    # ラベルマップ作成 (ソートしてIDを振る)
    sorted_labels = sorted(list(unique_labels))
    label_map = {label: i for i, label in enumerate(sorted_labels)}
    
    print(f"Found {len(unique_labels)} unique classes (excluding D0X).")
    print(f"Label map: {label_map}")
    
    return video_annotations, label_map

def extract_landmarks_from_video(video_path):
    """
    動画から全フレームのランドマークを抽出する。
    戻り値: (TotalFrames, 21, 3) のNumPy配列
    """
    mp_hands = mp.solutions.hands
    
    # 検出されなかった場合の補完用に前回の結果を保持
    last_landmarks = np.zeros((21, 3))
    
    landmarks_list = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
        
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, # 片手のみ想定
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # MediaPipeはRGBを期待
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            
            frame_landmarks = np.zeros((21, 3))
            
            if results.multi_hand_landmarks:
                # 検出された場合
                # 複数検出された場合は最初の手を使う（max_num_hands=1なので通常は1つ）
                hand_landmarks = results.multi_hand_landmarks[0]
                
                for i, lm in enumerate(hand_landmarks.landmark):
                    frame_landmarks[i] = [lm.x, lm.y, lm.z]
                
                last_landmarks = frame_landmarks.copy()
            else:
                # 検出されなかった場合は前回の値をコピー
                frame_landmarks = last_landmarks.copy()
                
            landmarks_list.append(frame_landmarks)
            
    cap.release()
    return np.array(landmarks_list)

def resample_sequence(sequence, target_len):
    """
    シーケンスを指定された長さにリサンプリング（線形補間）する。
    sequence: (T, 21, 3)
    return: (target_len, 21, 3)
    """
    current_len = len(sequence)
    if current_len == target_len:
        return sequence
    
    # 補間元のインデックス
    x = np.arange(current_len)
    # 補間先のインデックス
    new_x = np.linspace(0, current_len - 1, target_len)
    
    new_sequence = np.zeros((target_len, 21, 3))
    
    # 各ランドマーク(21)の各座標(3)について補間
    # 効率化のために軸を入れ替えてまとめて処理することも可能だが、
    # わかりやすさ優先でループ処理（NumPyのベクトル化で十分高速）
    for i in range(21):
        for j in range(3):
            new_sequence[:, i, j] = np.interp(new_x, x, sequence[:, i, j])
            
    return new_sequence

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # アノテーション読み込み
    video_anns, label_map = load_annotations()
    if not video_anns:
        print("No annotations found.")
        return

    # ラベルマップ保存
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
        
    all_data = []
    all_labels = []
    
    # 動画ファイルを検索
    video_files = glob.glob(os.path.join(VIDEO_DIR, f'*{VIDEO_EXT}'))
    print(f"Found {len(video_files)} video files in {VIDEO_DIR}.")
    
    processed_count = 0
    
    for video_path in video_files:
        # ファイル名（拡張子なし）を取得
        basename = os.path.splitext(os.path.basename(video_path))[0]
        
        # アノテーションがあるか確認
        if basename not in video_anns:
            continue
            
        print(f"Processing {basename}...")
        
        # 動画から全ランドマーク抽出
        full_landmarks = extract_landmarks_from_video(video_path)
        if full_landmarks is None or len(full_landmarks) == 0:
            print(f"Failed to extract landmarks from {basename}")
            continue
            
        total_frames = len(full_landmarks)
        
        # アノテーションに従って切り出し
        for ann in video_anns[basename]:
            label_name = ann['label']
            start = ann['start']
            end = ann['end']
            
            # フレーム範囲チェック (1-based index -> 0-based index)
            # startは1始まりなので、indexは start-1
            # endも含むので、スライスは end まで
            start_idx = max(0, start - 1)
            end_idx = min(total_frames, end)
            
            if start_idx >= end_idx:
                print(f"Invalid frame range: {start}-{end} (Total: {total_frames})")
                continue
                
            segment = full_landmarks[start_idx:end_idx]
            
            # リサンプリング
            resampled_segment = resample_sequence(segment, TARGET_FRAMES)
            
            all_data.append(resampled_segment)
            all_labels.append(label_map[label_name])
            
        processed_count += 1
        # 進捗表示
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} videos. Total samples so far: {len(all_data)}")

    if not all_data:
        print("No data generated.")
        return

    # NumPy配列に変換
    X = np.array(all_data, dtype=np.float32) # (N, TARGET_FRAMES, 21, 3)
    y = np.array(all_labels, dtype=np.int64) # (N,)
    
    print(f"Generation complete.")
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # 保存
    np.save(os.path.join(OUTPUT_DIR, 'real_landmarks.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'real_labels.npy'), y)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
