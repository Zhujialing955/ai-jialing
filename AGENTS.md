# AGENTS.md 项目固定规则，Codex启动自动读取
## 项目简介
本项目为GFTransMamba光场图像超分辨率重建，基于Mamba+Transformer混合网络，GAAM、CVESA、FREM为核心模块，Python+Pytorch。
## 代码规范
1. 修改代码前，先查看git状态，新建独立分支开发，不直接在main主分支改动。
2. 改动完成后，运行训练测试脚本验证，输出PSNR/SSIM指标。
3. 函数不要过长，超过40行要拆分；增加必要注释，论文实验代码要保留参数注释。
4. 禁止提交数据集、大权重文件、.env密钥、日志缓存到git。
## 项目目录说明
- /model：GAAM/CVESA/FREM/GFTransMamba网络模块
- /datasets：数据加载脚本
- /train.py：训练入口
- /test.py：测试指标计算入口
- /utils：损失函数、EPI绘图、工具函数
## 验收标准
修改代码后必须运行test.py，输出重建指标，对比修改前后PSNR/SSIM变化，给出diff变更说明。
