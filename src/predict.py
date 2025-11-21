import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from model import GestureTransformer
import random
import os


class PredictionViewer:
    def __init__(self, model, data, labels, classes):
        self.model = model
        self.data = data
        self.labels = labels
        self.classes = classes
        self.device = next(model.parameters()).device
        
        self.current_idx = 0
        self.num_samples = len(data)
        
        # グラフ設定
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.setup_plot()
        self.predict_and_show(random.randint(0, self.num_samples - 1))
        
    def setup_plot(self):
        self.scatters = self.ax.scatter([], [], [], c='r', marker='o')
        # MediaPipe Hand Connections (visualize.pyと同じ定義が必要だが、ここでは簡易的にimport元から取得するか再定義)
        # visualize.pyからHAND_CONNECTIONSをimport済みと仮定
        from visualize import HAND_CONNECTIONS
        self.lines = [self.ax.plot([], [], [], c='b')[0] for _ in HAND_CONNECTIONS]
        self.connections = HAND_CONNECTIONS

    def predict_and_show(self, idx):
        self.current_idx = idx
        sample_landmarks = self.data[idx]
        true_label = self.labels[idx]
        
        # 特徴量計算
        coords = sample_landmarks.reshape(sample_landmarks.shape[0], -1)
        velocity = np.zeros_like(coords)
        velocity[1:] = coords[1:] - coords[:-1]
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        features = np.concatenate([coords, velocity, acceleration], axis=-1)
        
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
        self.current_seq = sample_landmarks
        
        # クラス名取得（範囲外チェック）
        true_class = self.classes[true_label] if true_label < len(self.classes) else f"Unknown({true_label})"
        pred_class = self.classes[pred_idx] if pred_idx < len(self.classes) else f"Unknown({pred_idx})"
        
        self.pred_info = {
            'true': true_class,
            'pred': pred_class,
            'conf': probs[pred_idx].item(),
            'probs': probs
        }
        
        # コンソール出力
        print(f"\nSample {idx}: True=[{self.pred_info['true']}], Pred=[{self.pred_info['pred']}] ({self.pred_info['conf']:.2f})")
        
        # アニメーション再開
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.stop()
            
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(sample_landmarks), interval=50, blit=False, cache_frame_data=False
        )
        self.fig.canvas.draw_idle()

    def update(self, frame_cnt):
        seq_len = len(self.current_seq)
        idx = frame_cnt % seq_len
        frame_data = self.current_seq[idx]
        
        self.scatters._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
        
        for line, (start, end) in zip(self.lines, self.connections):
            x = [frame_data[start, 0], frame_data[end, 0]]
            y = [frame_data[start, 1], frame_data[end, 1]]
            z = [frame_data[start, 2], frame_data[end, 2]]
            line.set_data(x, y)
            line.set_3d_properties(z)
            
        # タイトル情報
        info = self.pred_info
        color = 'green' if info['true'] == info['pred'] else 'red'
        title = (f"Sample: {self.current_idx}\n"
                 f"True: {info['true']} | Pred: {info['pred']} ({info['conf']:.1%})\n"
                 f"[Right/Left] Change Sample")
        
        self.ax.set_title(title, color=color)
        
        # 軸固定
        self.ax.set_xlim(-0.2, 1.2)
        self.ax.set_ylim(-0.5, 1.5)
        self.ax.set_zlim(-0.5, 0.5)
        
        return self.scatters, *self.lines

    def on_key(self, event):
        if event.key == 'right':
            new_idx = (self.current_idx + 1) % self.num_samples
            self.predict_and_show(new_idx)
        elif event.key == 'left':
            new_idx = (self.current_idx - 1) % self.num_samples
            self.predict_and_show(new_idx)

def predict():
    import json
    
    # 設定
    INPUT_DIM = 189
    
    # 実データを使用
    data_path = 'data/real_landmarks.npy'
    label_path = 'data/real_labels.npy'
    label_map_path = 'data/label_map.json'
    
    # クラス定義読み込み
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            # 値(ID)でソートして名前のリストを作成
            sorted_items = sorted(label_map.items(), key=lambda x: x[1])
            CLASSES = [item[0] for item in sorted_items]
            NUM_CLASSES = len(CLASSES)
            print(f"Loaded {NUM_CLASSES} classes.")
    else:
        print("Label map not found. Using default.")
        NUM_CLASSES = 13 # デフォルト
        CLASSES = [str(i) for i in range(NUM_CLASSES)]
    
    # モデルのロード
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = GestureTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
    
    try:
        model.load_state_dict(torch.load("gesture_model.pth", map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please run train.py first.")
        return
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("The model architecture might have changed (e.g. number of classes). Please retrain.")
        return

    model.eval()
    
    # データロード
    if not os.path.exists(data_path):
        print("Data not found.")
        return
        
    raw_data = np.load(data_path)
    labels = np.load(label_path)
    
    print("Starting Prediction Viewer...")
    print("Press Right/Left arrow keys to switch samples.")
    
    viewer = PredictionViewer(model, raw_data, labels, CLASSES)
    plt.show()

if __name__ == "__main__":
    predict()
