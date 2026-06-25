#!/bin/bash

echo "========================================"
echo "  环境复原脚本 - Nanonets-OCR2-3B"
echo "  基础环境: PyTorch 2.7.0 + CUDA 11.8"
echo "========================================"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "检查 Python 版本..."
python --version

echo "检查 CUDA 版本..."
nvcc --version

echo "安装 PyTorch 2.7.0 (CUDA 11.8)..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

echo "安装 Transformers 库..."
pip install transformers==4.45.0

echo "安装 Accelerate 库..."
pip install accelerate==0.35.0

echo "安装 SentencePiece..."
pip install sentencepiece==0.2.0

echo "安装 Tokenizers..."
pip install tokenizers==0.20.0

echo "安装 Pillow (图像处理)..."
pip install Pillow==10.2.0

echo "安装 FastAPI 相关依赖..."
pip install fastapi==0.115.0
pip install uvicorn==0.30.1
pip install pydantic==2.9.1

echo "安装其他常用依赖..."
pip install numpy==1.26.4
pip install scipy==1.11.4
pip install scikit-learn==1.3.2

echo "========================================"
echo "  环境安装完成！"
echo "========================================"

echo "验证 PyTorch CUDA 支持..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 设备数: {torch.cuda.device_count()}')"

echo "验证模型推理依赖..."
python -c "from transformers import pipeline; print('Transformers pipeline 导入成功')"
python -c "from PIL import Image; print('Pillow 导入成功')"
python -c "from fastapi import FastAPI; print('FastAPI 导入成功')"