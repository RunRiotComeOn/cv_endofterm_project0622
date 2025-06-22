# 任务 2：基于 3D Gaussian Splatting 的物体重建和新视图合成

本仓库包含了“基于 3D Gaussian Splatting 的物体重建和新视图合成”项目的实现和实验结果。实验遵循任务 2 的要求，包括物体重建、新视图合成以及与 NeRF 方法的性能比较。本实验使用官方 [3D Gaussian Splatting 代码库](https://github.com/graphdeco-inria/gaussian-splatting) 和 COLMAP 进行相机参数估计。

## 概述
实验目标是使用 3D Gaussian Splatting 从多角度图像/视频重建 3D 物体，沿新轨迹渲染视频，并将其性能与原始 NeRF 和加速 NeRF 方法进行比较。

## 环境要求
- Python 3.8+
- 支持 CUDA 的 GPU
- 环境配置：
  - 先将目标仓库 clone 至本地
  ```bash
  # HTTPS
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
  ```
  - 再安装对应环境
  ```bash
  cd gaussian-splatting
  conda env create --file environment.yml
  conda activate gaussian_splatting
  ```
- 如果需要调整图像大小，需安装 [ImageMagick](https://imagemagick.org/)。
- 安装 [COLMAP](https://colmap.github.io/install.html) 用于相机位姿估计。
- 安装 [FFmpeg](https://ffmpeg.org/download.html) 用于视频转帧。

## 数据集准备
1. 拍摄物体的多角度视频（例如 `toy.mp4`）。
2. 创建对应数据集文件夹，并将对应视频 `toy.mp4` 移至该文件夹中
   ```bash
   mkdir dataset
   cd dataset
   ```
2. 使用 FFmpeg` 将视频分解为图像帧：
   ```bash
   ffmpeg -i toy1.mp4 -r 2 -q:v 2 ./images/frame_%04d.jpg
   ```
   或调整帧率：
   ```bash
   ffmpeg -i toy1.mp4 -vf "setpts=0.3*PTS" ./images/frame_%04d.jpg
   ```
3. 使用 COLMAP` 估计相机参数：
   ```bash
   cd 
   colmap automatic_reconstructor \
       --workspace_path dataset \
       --image_path dataset/images \
       --sparse 1 \
       --dense 0 \
       --camera_model SIMPLE_PINHOLE
   ```

4. 或者采用自带python文件（可选）：
   ```bash
   python convert.py -s dataset [--resize]
   ```

## 训练
先替换 `train.py` 文件为本仓库中的 `train.py`，再运行训练命令。
运行以下命令：
```bash
python /root/gaussian-splitting/train.py -s dataset --model_path dataset/models/
```
其中 `dataset` 替换为自己的数据集路径。

或者采用预编写好的 shell 文件
```bash
run.sh
```
- 完整训练指令：适用于高性能 GPU（≥24GB VRAM），追求最佳渲染质量
   ```bash
   python train.py -s <dataset_dir> -m <output_dir> --eval --iterations 30000 --densify_until_iter 15000 --densification_interval 100 --densify_grad_threshold 0.0002 --resolution 1.0
   ```
   - **参数说明**：
   - `-s <dataset_dir>`：数据集根目录（包含 `images/` 和 `sparse/`）。
   - `-m <output_dir>`：模型输出目录。
   - `--eval`：启用训练/测试分割，保留测试集用于评估。
   - `--iterations 30000`：总迭代次数，生成精细点云。
   - `--densify_until_iter 15000`：稠密化迭代次数，控制点云增长。
   - `--densification_interval 100`：稠密化频率，较小值增加点云密度。
   - `--densify_grad_threshold 0.0002`：点云分割梯度阈值，较小值保留更多细节。
   - `--resolution 1.0`：原始图像分辨率，保留最大细节。
   - **适用场景**：影视、虚拟现实等需要高质量渲染的场景。
   - **输出**：生成 `point_cloud/iteration_7000/` 和 `point_cloud/iteration_30000/`。

## 测试与渲染
1. （如果是用远程连接ssh的平台，需要在本地安装[该文件](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip)）使用训练好的模型渲染新视图：
   ```bash
   .\bin\SIBR_gaussianViewer_app.exe -m .\models
   ```
2. 渲染环绕物体的视频并在预留测试集上评估 PSNR 等指标。
   ```bash
   python render.py -m <output_dir> [--white_background] [--sh_degree <int>]
   ```
   - **参数说明**：
     - `-m <output_dir>`：训练模型的输出目录（包含 `point_cloud/iteration_XXXX/`）。
     - `--white_background`（可选）：强制白色背景，适合某些数据集（如室内场景）。
     - `--sh_degree <int>`（默认 3）：球谐函数阶数，控制光照复杂度。降低到 1 或 2 可减少内存占用，但可能影响光照效果。
   - **输出**：渲染图像保存在 `<output_dir>/train/` 和 `<output_dir>/test/`（如果训练时使用 `--eval`）。
   - **示例**：
     ```bash
     python render.py -m /path/to/output --white_background --sh_degree 2
     ```
3. 评估
   计算渲染图像的误差指标（如 PSNR、SSIM、LPIPS）：
   ```bash
   python metrics.py -m <output_dir>
   ```
   - **参数说明**：
   - `-m <output_dir>`：训练模型的输出目录。
   - **前提**：训练时需使用 `--eval` 保留测试集图像。
   - **输出**：误差指标保存在 `<output_dir>/metrics.json`。
   - **示例**：
      ```bash
      python metrics.py -m /path/to/output
      ```

## 常见问题
- 参考以下 issue 解决可能遇到的问题：
  - [Issue #292](https://github.com/graphdeco-inria/gaussian-splatting/issues/292)
  - [Issue #961](https://github.com/graphdeco-inria/gaussian-splatting/issues/961)

## 运行说明
1. 克隆本仓库：
   ```bash
   git clone https://github.com/RunRiotComeOn/cv_endofterm_project0622.git
   ```
2. 按照“环境要求”安装依赖。
3. 按“数据集准备”处理数据。
4. 运行“训练”命令进行模型训练。
5. 使用“测试与渲染”命令生成新视图或视频。

