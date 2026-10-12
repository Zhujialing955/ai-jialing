#!/bin/bash
# Codex云端环境初始化脚本（Linux容器）
apt update && apt install -y git wget unzip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.3.0 torchvision==0.18.0
pip install mamba-ssm timm numpy opencv-python scikit-image matplotlib tensorboard pandas
