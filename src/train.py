import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HandGestureDataset
from model import GestureTransformer
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt

def plot_history(history, save_path='training_history.png'):
    """学習履歴をグラフ化して保存"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")
    # plt.show() # サーバー環境等でGUIがない場合はコメントアウト推奨

def train():
    parser = argparse.ArgumentParser(description='Train Gesture Transformer')
    parser.add_argument('--resume', action='store_true', help='Resume training from existing model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    args = parser.parse_args()

    # ハイパーパラメータ設定
    BATCH_SIZE = 16
    EPOCHS = args.epochs
    LEARNING_RATE = 0.001
    MODEL_PATH = "gesture_model.pth"
    
    # 特徴量次元数: (21 landmarks * 3 coords) * 3 (pos + vel + acc)
    INPUT_DIM = 189 
    
    # 実データを使用
    data_path = 'data/real_landmarks.npy'
    label_path = 'data/real_labels.npy'
    label_map_path = 'data/label_map.json'
    
    # クラス数を自動取得
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            NUM_CLASSES = len(label_map)
            print(f"Detected {NUM_CLASSES} classes from {label_map_path}")
    else:
        print(f"Label map not found at {label_map_path}. Using default 13.")
        NUM_CLASSES = 13

    # データ生成確認
    if not os.path.exists(data_path):
        print("Data not found. Please run 'python src/process_real_data.py' first.")
        return

    # データセットとデータローダー
    dataset = HandGestureDataset(data_path, label_path)
    
    # 訓練データと検証データに分割 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples...")

    # モデル構築
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = GestureTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
    
    # モデルのロード（追加学習モード）
    if args.resume:
        if os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print(f"Resumed training from {MODEL_PATH}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Starting training from scratch.")
        else:
            print(f"Model file {MODEL_PATH} not found. Starting training from scratch.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 履歴記録用
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 学習ループ
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 精度計算用
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        # 検証ループ
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_avg_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} Acc: {accuracy:.2f}% | Val Loss: {val_avg_loss:.4f} Acc: {val_accuracy:.2f}%")
        
        # 履歴に追加
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(accuracy)
        history['val_loss'].append(val_avg_loss)
        history['val_acc'].append(val_accuracy)

    print("Training finished.")
    
    # モデル保存
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # グラフ保存
    plot_history(history)

if __name__ == "__main__":
    train()
