import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
import random

# 超参数配置
class Config:
    def __init__(self):
        # 数据相关
        self.batch_size = 64
        self.num_classes = 3
        
        # 模型相关
        self.hidden_dims = [512, 256, 64]  # 隐藏层维度
        self.dropout_rate = 0.25
        
        # LSTM相关
        self.lstm_hidden_size = 512  # LSTM隐藏层大小
        self.lstm_num_layers = 1     # LSTM层数
        self.lstm_bidirectional = True  # 是否使用双向LSTM
        self.fusion_strategy = 'attention'   # 融合策略: 'last', 'mean', 'attention'
        self.use_original_sequence = False  # 是否使用原始时间序列数据

        # CBAM相关
        self.use_cbam = True  # 是否使用CBAM模块
        self.cbam_reduction_ratio = 16  # CBAM通道注意力压缩比例

        # 训练相关
        self.num_epochs = 20
        self.learning_rate = 0.0001
        self.weight_decay = 1e-6
        self.seed = 1500
        
        # 学习率调度相关
        if self.learning_rate == 0.0001:
            self.min_lr = 1e-6  # 最小学习率
        elif self.learning_rate == 0.001:
            self.min_lr = 1e-5  # 最小学习率
        elif self.learning_rate == 0.00001:
            self.min_lr = 1e-7  # 最小学习率

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# CBAM模块 - 适配时间序列数据
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channels = channels
        
        # 通道注意力模块
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        
        # 空间注意力模块（针对时间序列，我们使用时间注意力）
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        # x的形状: (batch_size, channels, seq_len)
        # 应用通道注意力
        x = self.channel_attention(x) * x
        
        # 应用空间（时间）注意力
        x = self.spatial_attention(x) * x
        
        return x

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x的形状: (batch_size, channels, seq_len)
        b, c, s = x.size()
        
        # 全局平均池化和最大池化
        avg_out = self.avg_pool(x).view(b, c)  # (batch_size, channels)
        max_out = self.max_pool(x).view(b, c)  # (batch_size, channels)
        
        # 通过全连接层
        avg_out = self.fc(avg_out)  # (batch_size, channels)
        max_out = self.fc(max_out)  # (batch_size, channels)
        
        # 相加并通过sigmoid
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1)  # (batch_size, channels, 1)
        
        return out

# 空间注意力模块（针对时间序列）
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x的形状: (batch_size, channels, seq_len)
        # 在通道维度上计算平均和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, seq_len)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, seq_len)
        
        # 拼接平均和最大特征
        x_concat = torch.cat([avg_out, max_out], dim=1)  # (batch_size, 2, seq_len)
        
        # 通过卷积层
        out = self.conv(x_concat)  # (batch_size, 1, seq_len)
        out = self.sigmoid(out)  # (batch_size, 1, seq_len)
        
        return out

# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, feature_dim, config):
        super(Classifier, self).__init__()
        
        # CBAM模块（如果启用）
        if config.use_cbam:
            self.cbam = CBAM(channels=feature_dim, reduction_ratio=config.cbam_reduction_ratio)
        
        # LSTM层用于融合多尺度特征
        self.lstm = nn.LSTM(
            input_size=feature_dim,  # 每个尺度的特征维度 (2048)
            hidden_size=config.lstm_hidden_size,  # LSTM隐藏层维度
            num_layers=config.lstm_num_layers,  # LSTM层数
            batch_first=True,
            dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0,
            bidirectional=config.lstm_bidirectional  # 双向LSTM
        )
        
        # 计算LSTM输出的维度
        lstm_output_dim = config.lstm_hidden_size * (2 if config.lstm_bidirectional else 1)
        
        # 注意力机制（如果使用attention融合策略）
        if config.fusion_strategy == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_dim // 2, 1)
            )
        
        # 全连接分类层，支持任意层数
        layers = []
        prev_dim = lstm_output_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=config.dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, config.num_classes))
        self.classifier = nn.Sequential(*layers)
        
        self.fusion_strategy = config.fusion_strategy
        self.config = config
    
    def forward(self, x):
        # x的形状: (batch_size, 3, 45, 2048)
        batch_size, num_scales, seq_len, feature_dim = x.shape
        
        if self.config.use_original_sequence:
            # 使用原始时间序列数据
            # 重塑为 (batch_size * 3, 45, 2048)
            x_reshaped = x.view(batch_size * num_scales, seq_len, feature_dim)
            
            # 对每个尺度的特征应用CBAM（如果启用）
            if self.config.use_cbam:
                # 重塑为 (batch_size * 3, 2048, 45) 用于CBAM
                x_for_cbam = x_reshaped.transpose(1, 2)  # (batch_size * 3, 2048, 45)
                x_for_cbam = self.cbam(x_for_cbam)  # 应用CBAM
                x_reshaped = x_for_cbam.transpose(1, 2)  # 转回 (batch_size * 3, 45, 2048)
            
            # 通过LSTM处理每个尺度的序列
            lstm_out, (hidden, cell) = self.lstm(x_reshaped)  # (batch_size * 3, 45, hidden_size*2)
            
            # 取每个序列的最后一个时间步
            scale_features = lstm_out[:, -1, :]  # (batch_size * 3, hidden_size*2)
            
            # 重塑回 (batch_size, 3, hidden_size*2)
            scale_features = scale_features.view(batch_size, num_scales, -1)
            
            # 根据融合策略融合不同尺度的特征
            if self.fusion_strategy == 'last':
                fused_feature = scale_features[:, -1, :]  # 取最后一个尺度
            
            elif self.fusion_strategy == 'mean':
                fused_feature = scale_features.mean(dim=1)  # 对所有尺度取平均
            
            elif self.fusion_strategy == 'attention':
                # 使用注意力机制融合不同尺度
                attention_weights = self.attention(scale_features)  # (batch_size, 3, 1)
                attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, 3, 1)
                fused_feature = torch.sum(scale_features * attention_weights, dim=1)  # (batch_size, hidden_size*2)
            
            else:
                raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        else:
            # 使用平均特征（原来的方法）
            # 重塑为LSTM输入格式: (batch_size, 3, 2048)
            # 我们对每个尺度的45个时间步取平均，得到每个尺度的代表特征
            x = x.mean(dim=2)  # (batch_size, 3, 2048)
            
            # 对每个尺度的特征应用CBAM（如果启用）
            if self.config.use_cbam:
                # 重塑为 (batch_size, 2048, 3) 用于CBAM
                x_for_cbam = x.transpose(1, 2)  # (batch_size, 2048, 3)
                x_for_cbam = self.cbam(x_for_cbam)  # 应用CBAM
                x = x_for_cbam.transpose(1, 2)  # 转回 (batch_size, 3, 2048)
            
            # 通过LSTM处理多尺度特征序列
            lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch_size, 3, hidden_size*2)
            
            # 根据融合策略选择不同的融合方法
            if self.fusion_strategy == 'last':
                # 取最后一个时间步的输出作为融合特征
                fused_feature = lstm_out[:, -1, :]  # (batch_size, hidden_size*2)
            
            elif self.fusion_strategy == 'mean':
                # 对所有时间步取平均
                fused_feature = lstm_out.mean(dim=1)  # (batch_size, hidden_size*2)
            
            elif self.fusion_strategy == 'attention':
                # 使用注意力机制
                attention_weights = self.attention(lstm_out)  # (batch_size, 3, 1)
                attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, 3, 1)
                fused_feature = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_size*2)
            
            else:
                raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # 通过分类器
        output = self.classifier(fused_feature)
        
        return output

# 加载数据
def load_data():
    data_dir_512 = './data/datasets_npy'
    data_dir_256 = './data/datasets_256_npy'
    data_dir_128 = './data/datasets_128_npy'
    label_file = 'labels_dataset_45.csv'

    # 读取标签文件（有表头：folder_name, label）
    labels_df = pd.read_csv(label_file)
    labels_df = labels_df.rename(columns={'folder_name': 'ID', 'label': 'Label'})

    features = []
    labels = []

    # 读取特征文件
    for _, row in labels_df.iterrows():
        filename = str(row['ID'])
        label    = int(row['Label'])
        file_path_512 = os.path.join(data_dir_512, f"{filename}.npy")
        file_path_256 = os.path.join(data_dir_256, f"{filename}.npy")
        file_path_128 = os.path.join(data_dir_128, f"{filename}.npy")

        if all(os.path.exists(path) for path in [file_path_512, file_path_256, file_path_128]):
            feature_512 = np.load(file_path_512)
            feature_256 = np.load(file_path_256)
            feature_128 = np.load(file_path_128)

            multi_scale_feature = np.stack([feature_512, feature_256, feature_128], axis=0)

            features.append(multi_scale_feature)
            labels.append(label)

    return np.array(features), np.array(labels)

# 平衡数据集函数
def balance_dataset(features, labels):
    """
    根据0类和1类的数量差异，随机复制数量少的那一类，以保证两类训练数据一样
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n原始数据集类别分布:")
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 样本")
    
    # 找到数量最多的类别
    max_count = np.max(counts)
    print(f"\n目标平衡数量: {max_count} 样本/类")
    
    balanced_features = []
    balanced_labels = []
    
    for label in unique_labels:
        # 获取当前类别的所有样本
        label_indices = np.where(labels == label)[0]
        label_features = features[label_indices]
        label_count = len(label_indices)
        
        if label_count < max_count:
            # 如果当前类别样本数量少于最大数量，需要复制样本
            # 计算需要复制的样本数量
            need_copy = max_count - label_count
            print(f"类别 {label}: 需要复制 {need_copy} 个样本")
            
            # 随机选择要复制的样本索引
            copy_indices = np.random.choice(label_indices, size=need_copy, replace=True)
            
            # 添加原始样本
            balanced_features.extend(label_features)
            balanced_labels.extend([label] * label_count)
            
            # 添加复制的样本
            balanced_features.extend(features[copy_indices])
            balanced_labels.extend([label] * need_copy)
        else:
            # 如果当前类别样本数量已经足够，直接添加
            balanced_features.extend(label_features)
            balanced_labels.extend([label] * label_count)
    
    balanced_features = np.array(balanced_features)
    balanced_labels = np.array(balanced_labels)
    
    # 验证平衡结果
    unique_labels_balanced, counts_balanced = np.unique(balanced_labels, return_counts=True)
    print(f"\n平衡后数据集类别分布:")
    for label, count in zip(unique_labels_balanced, counts_balanced):
        print(f"类别 {label}: {count} 样本")
    
    return balanced_features, balanced_labels

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), correct / total

# 验证函数
def validate_model(model, val_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    return all_preds, all_labels, all_scores, total_loss / len(val_loader), correct / total

def main():
    # 初始化配置
    config = Config()
    # set_seed(config.seed)
    
    # 加载数据
    features, labels = load_data()
    
    # 平衡数据集
    # features, labels = balance_dataset(features, labels)
    
    # 转换为PyTorch张量
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用StratifiedKFold进行五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)

    fold_accuracies       = []
    fold_macro_precision  = []
    fold_macro_recall     = []
    fold_macro_f1         = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\n======== Fold {fold + 1}/5 ========")

        # 准备数据
        train_features, val_features = features[train_idx], features[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_dataset = CustomDataset(train_features, train_labels)
        val_dataset   = CustomDataset(val_features, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size)

        # 初始化模型
        feature_dim = features.shape[-1]
        model = Classifier(feature_dim=feature_dim, config=config).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=config.min_lr
        )

        # 训练模型，只记录最后一个 epoch
        last_epoch_preds  = None
        last_epoch_labels = None
        last_epoch_scores = None
        last_epoch_acc    = 0

        for epoch in range(config.num_epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_preds, val_labels, val_scores, val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f'Epoch {epoch+1}/{config.num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')

            if epoch == config.num_epochs - 1:
                last_epoch_preds  = val_preds
                last_epoch_labels = val_labels
                last_epoch_scores = val_scores
                last_epoch_acc    = val_acc

        fold_accuracies.append(last_epoch_acc)
        precision, recall, f1, _ = precision_recall_fscore_support(
            last_epoch_labels, last_epoch_preds, average='macro', zero_division=0)
        fold_macro_precision.append(precision)
        fold_macro_recall.append(recall)
        fold_macro_f1.append(f1)

        # 保存预测概率供后续 ROC 使用
        os.makedirs('fold_predictions/Ours_副本', exist_ok=True)
        np.save(f'fold_predictions/Ours_副本/fold{fold+1}_y_true.npy',  np.array(last_epoch_labels))
        np.save(f'fold_predictions/Ours_副本/fold{fold+1}_y_score.npy', np.array(last_epoch_scores))

        print(f"\nFold {fold + 1} Last Epoch Val Acc: {last_epoch_acc:.4f}")
        print(classification_report(last_epoch_labels, last_epoch_preds, digits=3))
        print("-" * 50)

    # 输出五折汇总结果
    print("\n五折交叉验证最终结果:")
    print(f"平均验证集准确率: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"平均 Macro Precision: {np.mean(fold_macro_precision):.4f}")
    print(f"平均 Macro Recall:    {np.mean(fold_macro_recall):.4f}")
    print(f"平均 Macro F1-score:  {np.mean(fold_macro_f1):.4f}")

if __name__ == '__main__':
    main()
