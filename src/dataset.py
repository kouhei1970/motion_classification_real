import torch
from torch.utils.data import Dataset
import numpy as np

class HandGestureDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path) # (N, T, 21, 3)
        self.labels = np.load(label_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # (T, 21, 3)
        landmarks = self.data[idx]
        
        # 特徴量計算: 座標(63) + 速度(63) + 加速度(63) = 189次元
        
        # 1. 座標 (T, 63)
        coords = landmarks.reshape(landmarks.shape[0], -1)
        
        # 2. 速度 (T, 63)
        # v_t = p_t - p_{t-1}
        # 先頭フレームの速度は0とする
        velocity = np.zeros_like(coords)
        velocity[1:] = coords[1:] - coords[:-1]
        
        # 3. 加速度 (T, 63)
        # a_t = v_t - v_{t-1}
        # 先頭フレームの加速度は0とする
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        
        # 結合 (T, 189)
        features = np.concatenate([coords, velocity, acceleration], axis=-1)
        
        return torch.FloatTensor(features), torch.tensor(self.labels[idx], dtype=torch.long)
