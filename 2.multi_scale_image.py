import os
from PIL import Image
import shutil
from tqdm import tqdm

def create_directory_structure(src_dir, dst_dir):
    """创建目标目录结构"""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    
    # 复制目录结构
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        if rel_path == '.':
            continue
        os.makedirs(os.path.join(dst_dir, rel_path), exist_ok=True)

def resize_image(src_path, dst_path, size):
    """调整图像大小并保持质量"""
    try:
        with Image.open(src_path) as img:
            # 使用LANCZOS重采样方法以获得最佳质量
            resized_img = img.resize(size, Image.Resampling.LANCZOS)
            # 保存时使用最高质量
            resized_img.save(dst_path, quality=95, optimize=True)
    except Exception as e:
        print(f"处理图像 {src_path} 时出错: {str(e)}")

def process_dataset(src_dir, dst_dir_256, dst_dir_128):
    """处理整个数据集"""
    # 创建目标目录结构
    create_directory_structure(src_dir, dst_dir_256)
    create_directory_structure(src_dir, dst_dir_128)
    
    # 获取所有图像文件
    image_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append((root, file))
    
    # 使用tqdm显示进度
    for root, file in tqdm(image_files, desc="处理图像"):
        rel_path = os.path.relpath(root, src_dir)
        src_path = os.path.join(root, file)
        
        # 处理256x256版本
        dst_path_256 = os.path.join(dst_dir_256, rel_path, file)
        resize_image(src_path, dst_path_256, (256, 256))
        
        # 处理128x128版本
        dst_path_128 = os.path.join(dst_dir_128, rel_path, file)
        resize_image(src_path, dst_path_128, (128, 128))

if __name__ == "__main__":
    src_dir = "./data/datasets"
    dst_dir_256 = "./data/datasets_256"
    dst_dir_128 = "./data/datasets_128"
    
    print("开始处理数据集...")
    process_dataset(src_dir, dst_dir_256, dst_dir_128)
    print("处理完成！")
