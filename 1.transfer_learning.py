import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载预训练的 ResNet-50，并去掉最后的全连接层
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的分类层
model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整到固定尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        return image, img_path

# 获取所有子文件夹
data_root = "./data/datasets"
result_root = "./data/datasets_npy"
# 获取所有子文件夹，并按数字从小到大排序
sub_dirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))], key=int)

os.makedirs(result_root, exist_ok=True)  # 确保输出目录存在

for sub_dir in sub_dirs:
    input_folder = os.path.join(data_root, sub_dir)
    
    # 获取当前子文件夹下的所有 jpg 图像
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    # 创建数据加载器
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 用于存储当前文件夹所有图像的特征
    all_features = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"Processing {sub_dir}"):
            images = images.to(device)
            features = model(images)  # 提取特征
            features = features.view(features.size(0), -1).cpu().numpy()  # 变成 1D 向量
            all_features.append(features)
    
    # 将所有特征拼接成一个数组
    all_features = np.vstack(all_features)
    # print('------------', all_features.shape)
    
    # 确保特征数量为45
    if all_features.shape[0] != 45:
        print(f"警告：文件夹 {sub_dir} 中的图像数量不是45张，实际数量为 {all_features.shape[0]}")
    
    # 保存特征
    output_file = os.path.join(result_root, f"{sub_dir}.npy")
    np.save(output_file, all_features)
    print(f"已保存文件夹 {sub_dir} 的特征，形状为 {all_features.shape}")

print("所有特征提取完毕，保存在目录:", result_root)
