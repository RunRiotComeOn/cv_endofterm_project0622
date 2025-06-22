#!/bin/bash
# 训练命令 1：快速实验（低分辨率，少迭代）
python /root/gaussian-splatting/train.py -s ./dataset --model_path ./dataset/models/run1 --resolution 4 --iterations 10000 --densify_until_iter 5000 --densify_grad_threshold 0.0003 --eval --lambda_dssim 0.2

# 训练命令 2：平衡质量与速度（默认分辨率，中等迭代）
python /root/gaussian-splatting/train.py -s ./dataset --model_path ./dataset/models/run2 --resolution -1 --iterations 20000 --densify_until_iter 10000 --densify_grad_threshold 0.0002 --eval --lambda_dssim 0.3

# 训练命令 3：高质量重建（高分辨率，多迭代）
python /root/gaussian-splatting/train.py -s ./dataset --model_path ./dataset/models/run3 --resolution 1 --iterations 40000 --densify_until_iter 20000 --densify_grad_threshold 0.0001 --eval --lambda_dssim 0.4

# 训练命令 4：细节增强（中等分辨率，延长密度化）
python /root/gaussian-splatting/train.py -s ./dataset --model_path ./dataset/models/run4 --resolution 2 --iterations 30000 --densify_until_iter 15000 --densify_grad_threshold 0.0001 --eval --lambda_dssim 0.3