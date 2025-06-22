# NeRF & TensoRF 视频三维重建流程

## 项目概述
本项目可将您拍摄的普通视频转化为三维场景，使用以下技术：
- NeRF (神经辐射场)：实现照片级真实感的三维重建
- TensoRF：加速训练和渲染过程的改进方案

## 硬件要求
- GPU：NVIDIA A800 (或类似高性能显卡)
- CUDA 版本：11.3
- 显存：建议≥40GB以获得最佳效果

## 环境安装

### 1. 克隆所有必要的代码库：

```bash
# 创建项目目录
mkdir -p ~/autodl-tmp/nerf_project
cd ~/autodl-tmp/nerf_project

# 克隆 torch-ngp (Instant-NGP实现)
git clone https://github.com/ashawkey/torch-ngp.git
cd torch-ngp
pip install -r requirements.txt
cd ..

# 克隆 nerf-pytorch (原始NeRF实现)
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
cd ..

# 克隆 TensoRF (加速版NeRF)
git clone https://github.com/apchenstu/TensoRF.git
cd TensoRF
pip install -r requirements.txt
cd ..
```

- 注意：请用文件夹下的 `TensoRF` 中的文件替换克隆下的 `TensoRF` 中对应位置的同名文件，以顺利运行项目。

### 2. 安装系统依赖

```bash
# 安装COLMAP (三维重建工具)
sudo apt-get install colmap ffmpeg

# 安装PyTorch (CUDA 11.3版本)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## 项目目录结构
```
/
├── my_video/                # 输入视频和处理数据
│   ├── my_video.mp4         # 原始视频文件
│   ├── images/              # 提取的视频帧
│   ├── colmap_db/           # COLMAP数据库
│   ├── colmap_sparse/       # 稀疏重建结果
│   └── colmap_text/         # 文本格式输出
├── torch-ngp/               # Instant-NGP实现
├── nerf-pytorch/            # 原始NeRF实现
└── TensoRF/                 # TensoRF实现
```

## 项目流程
### 视频转图片帧

```bash
mkdir -p /root/autodl-tmp/my_video/images

# 转换为JPG格式(推荐)
ffmpeg -i /root/autodl-tmp/my_video.mp4 -r 2 -q:v 2 /root/autodl-tmp/my_video/images/frame_%04d.jpg

# 或转换为PNG格式
ffmpeg -i /root/autodl-tmp/my_video.mp4 -r 2 /root/autodl-tmp/my_video/images/frame_%04d.png
```

### 使用COLMAP估计相机参数

```bash
mkdir -p /root/autodl-tmp/my_video/colmap_db
mkdir -p /root/autodl-tmp/my_video/colmap_sparse

# 特征提取
colmap feature_extractor \
  --database_path /root/autodl-tmp/my_video/colmap_db/database.db \
  --image_path /root/autodl-tmp/my_video/images \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1

# 特征匹配
colmap exhaustive_matcher \
  --database_path /root/autodl-tmp/my_video/colmap_db/database.db

# 稀疏重建
colmap mapper \
  --database_path /root/autodl-tmp/my_video/colmap_db/database.db \
  --image_path /root/autodl-tmp/my_video/images \
  --output_path /root/autodl-tmp/my_video/colmap_sparse

# 转换为文本格式
mkdir -p /root/autodl-tmp/my_video/colmap_text
colmap model_converter \
  --input_path /root/autodl-tmp/my_video/colmap_sparse/0 \
  --output_path /root/autodl-tmp/my_video/colmap_text \
  --output_type TXT

# 转换为NeRF所需格式
python ~/autodl-tmp/torch-ngp/scripts/colmap2nerf.py \
  --images /root/autodl-tmp/my_video/images \
  --colmap_text /root/autodl-tmp/my_video/colmap_text \
  --hold 8  
# hold 8 自动划分数据集
```

### 训练NeRF模型

```bash
# 准备数据目录
mkdir -p /root/autodl-tmp/nerf-pytorch/data/my_video/
cp /root/autodl-tmp/my_video/transforms_train.json /root/autodl-tmp/nerf-pytorch/data/my_video/
cp /root/autodl-tmp/my_video/transforms_val.json /root/autodl-tmp/nerf-pytorch/data/my_video/
cp /root/autodl-tmp/my_video/transforms_test.json /root/autodl-tmp/nerf-pytorch/data/my_video/
cp -r /root/autodl-tmp/my_video/images /root/autodl-tmp/nerf-pytorch/data/my_video/

# 开始训练
cd /root/autodl-tmp/nerf-pytorch
python run_nerf.py --config configs/lego.txt --datadir data/my_video

# 仅渲染模式
python run_nerf.py --config configs/lego.txt --datadir data/my_video --render_only --render_path renders/my_video
```

### 训练TensoRF模型(加速版)
```bash
# 准备数据目录
mkdir -p /root/autodl-tmp/TensoRF/data/my_video/
cp /root/autodl-tmp/my_video/transforms_train.json /root/autodl-tmp/TensoRF/data/my_video/
cp /root/autodl-tmp/my_video/transforms_val.json /root/autodl-tmp/TensoRF/data/my_video/
cp /root/autodl-tmp/my_video/transforms_test.json /root/autodl-tmp/TensoRF/data/my_video/
cp -r /root/autodl-tmp/my_video/images /root/autodl-tmp/TensoRF/data/my_video/

# 开始训练
cd /root/autodl-tmp/TensoRF
mkdir -p /root/autodl-tmp/TensoRF/log
python train.py --config configs/lego.txt
```


## 常见问题解决

​​### COLMAP相关问题：​​

1. 如果特征提取失败，尝试降低图像分辨率
2. 确保CUDA 11.3正确安装以启用GPU加速
​​
### 显存不足问题：​​

1. 遇到OOM错误时减小批处理大小
2. A800显卡通常可以使用比默认值更大的批处理量
​​
### 依赖冲突问题：​​

请为 NeRF 和 TensoRF 创建独立的 conda 环境
