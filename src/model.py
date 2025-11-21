import torch
import torch.nn as nn
import math

class GestureTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        # 入力特徴量をTransformerの次元に埋め込む
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        # batch_first=True: 入力が (Batch, Seq, Feature) の形式であることを指定
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 分類ヘッド
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        
        # Embedding & Positional Encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer Encoder
        # output: (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src)
        
        # 集約 (Aggregation)
        # シーケンス全体の特徴をまとめるために平均プーリングを使用
        # (あるいは [CLS] トークンのようなものを使うか、最後のフレームを使う方法もある)
        output = output.mean(dim=1) 
        
        # Classification
        # (batch_size, num_classes)
        logits = self.classifier(output)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # pe: (max_len, 1, d_model) -> transpose to (1, max_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
