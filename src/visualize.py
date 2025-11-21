import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import sys

# MediaPipe Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def visualize_sequence(landmarks_seq, interval=50):
    """
    landmarks_seq: (seq_len, 21, 3) のnumpy配列
    interval: フレーム間の更新間隔(ms)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 軸の範囲設定（データに合わせて調整）
    # MediaPipeは通常0~1の範囲だが、シミュレーションデータに合わせて調整
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # プロット要素の初期化
    scatters = ax.scatter([], [], [], c='r', marker='o')
    lines = [ax.plot([], [], [], c='b')[0] for _ in HAND_CONNECTIONS]

    def update(frame_idx):
        frame_data = landmarks_seq[frame_idx] # (21, 3)
        
        # 散布図の更新
        scatters._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
        
        # 線の更新
        for line, (start, end) in zip(lines, HAND_CONNECTIONS):
            x = [frame_data[start, 0], frame_data[end, 0]]
            y = [frame_data[start, 1], frame_data[end, 1]]
            z = [frame_data[start, 2], frame_data[end, 2]]
            line.set_data(x, y)
            line.set_3d_properties(z)
            
        ax.set_title(f"Frame: {frame_idx}")
        return scatters, *lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(landmarks_seq), interval=interval, blit=False
    )
    
    plt.show()

class GestureDatasetViewer:
    def __init__(self, data_path, label_path=None, class_names=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path) if label_path and os.path.exists(label_path) else None
        self.class_names = class_names
        
        # クラスごとにデータを分類
        self.class_data_indices = {}
        if self.labels is not None and self.class_names:
            for i, name in enumerate(self.class_names):
                # そのクラスに該当するインデックスを取得
                indices = np.where(self.labels == i)[0]
                self.class_data_indices[name] = indices
        
        # 各クラスの現在の表示インデックス (クラス内でのオフセット)
        self.current_sample_offset = 0
        
        # 最小サンプル数を確認 (ページングの上限用)
        self.min_samples = min([len(indices) for indices in self.class_data_indices.values()]) if self.class_data_indices else 0
        
        print(f"Loaded {len(self.data)} samples total.")
        print(f"Min samples per class: {self.min_samples}")
        print("Controls: [Right Arrow] Next Sample, [Left Arrow] Previous Sample")

        # サブプロットの作成 (クラス数分)
        self.num_classes = len(self.class_names) if self.class_names else 1
        
        # ページング設定
        self.classes_per_page = 2
        self.current_page = 0
        self.total_pages = (self.num_classes + self.classes_per_page - 1) // self.classes_per_page
        
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 自動回転フラグ
        self.auto_rotate = False
        self.rotate_angle = 0
        
        self.setup_page()
        self.start_animation()

    def setup_page(self):
        self.fig.clf()
        self.axes = []
        self.scatters_list = []
        self.lines_list = []
        
        start_idx = self.current_page * self.classes_per_page
        end_idx = min(start_idx + self.classes_per_page, self.num_classes)
        
        # 1行2列で表示
        cols = 2
        rows = 1
        
        for i in range(start_idx, end_idx):
            # サブプロットの位置 (1 or 2)
            plot_idx = i - start_idx + 1
            ax = self.fig.add_subplot(rows, cols, plot_idx, projection='3d')
            self.setup_plot(ax, self.class_names[i])
            self.axes.append(ax)
            
            scatters = ax.scatter([], [], [], c='r', marker='o')
            lines = [ax.plot([], [], [], c='b')[0] for _ in HAND_CONNECTIONS]
            
            self.scatters_list.append(scatters)
            self.lines_list.append(lines)
            
        print(f"Showing page {self.current_page + 1}/{self.total_pages} (Classes {start_idx+1}-{end_idx})")
        print("Controls: [Right/Left] Sample, [Up/Down] Page, [R] Reset View, [A] Auto Rotate")
        
        # 操作説明を画面上に表示
        help_text = (
            "Controls:\n"
            "  [Right/Left] Change Sample\n"
            "  [Up/Down]    Change Page\n"
            "  [R]          Reset View\n"
            "  [A]          Auto Rotate"
        )
        self.fig.text(0.01, 0.01, help_text, fontsize=10, va='bottom', ha='left', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def setup_plot(self, ax, title):
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.5, 1.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        # 初期視点
        ax.view_init(elev=20, azim=-60)

    def update(self, frame_cnt):
        artists = []
        
        # 自動回転
        if self.auto_rotate:
            self.rotate_angle = (self.rotate_angle + 1) % 360
            for ax in self.axes:
                ax.view_init(elev=20, azim=self.rotate_angle)
        
        start_idx = self.current_page * self.classes_per_page
        end_idx = min(start_idx + self.classes_per_page, self.num_classes)
        
        for i in range(start_idx, end_idx):
            # リスト内のインデックス (0 or 1)
            list_idx = i - start_idx
            class_name = self.class_names[i]
            
            # そのクラスのデータリストを取得
            indices = self.class_data_indices[class_name]
            
            # 現在のオフセットに対応するデータインデックス
            data_idx = indices[self.current_sample_offset % len(indices)]
            
            sequence = self.data[data_idx]
            seq_len = len(sequence)
            
            # アニメーションフレーム
            frame_idx = frame_cnt % seq_len
            frame_data = sequence[frame_idx]
            
            # 描画更新
            scatters = self.scatters_list[list_idx]
            lines = self.lines_list[list_idx]
            ax = self.axes[list_idx]
            
            scatters._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
            artists.append(scatters)
            
            for line, (start, end) in zip(lines, HAND_CONNECTIONS):
                x = [frame_data[start, 0], frame_data[end, 0]]
                y = [frame_data[start, 1], frame_data[end, 1]]
                z = [frame_data[start, 2], frame_data[end, 2]]
                line.set_data(x, y)
                line.set_3d_properties(z)
                artists.append(line)
                
            # タイトルにサンプル番号を表示
            ax.set_title(f"{class_name} | Sample {self.current_sample_offset + 1}")

        return artists

    def on_key(self, event):
        if event.key == 'right':
            self.current_sample_offset += 1
        elif event.key == 'left':
            if self.current_sample_offset > 0:
                self.current_sample_offset -= 1
        elif event.key == 'down':
            # 次のページ
            self.current_page = (self.current_page + 1) % self.total_pages
            self.setup_page()
        elif event.key == 'up':
            # 前のページ
            self.current_page = (self.current_page - 1) % self.total_pages
            self.setup_page()
        elif event.key == 'r':
            # 視点リセット
            self.auto_rotate = False
            for ax in self.axes:
                ax.view_init(elev=20, azim=-60)
            self.fig.canvas.draw_idle()
        elif event.key == 'a':
            # 自動回転トグル
            self.auto_rotate = not self.auto_rotate
            if self.auto_rotate:
                # 現在の角度から開始
                self.rotate_angle = self.axes[0].azim
        
    def start_animation(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=None, interval=50, blit=False, cache_frame_data=False
        )
        plt.show()

if __name__ == "__main__":
    # デフォルト設定
    data_path = 'data/real_landmarks.npy'
    label_path = 'data/real_labels.npy'
    label_map_path = 'data/label_map.json'
    
    class_names = []

    print("Mode: Real Data")
    # ラベルマップの読み込み
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            # 値(ID)でソートして名前のリストを作成
            sorted_items = sorted(label_map.items(), key=lambda x: x[1])
            class_names = [item[0] for item in sorted_items]
        print(f"Loaded {len(class_names)} classes from {label_map_path}")
    else:
        print(f"Warning: {label_map_path} not found. Using default/empty class names.")

    if os.path.exists(data_path):
        viewer = GestureDatasetViewer(data_path, label_path, class_names=class_names)
    else:
        print(f"Data file not found: {data_path}")
        print("Please run 'python src/process_real_data.py' first.")
