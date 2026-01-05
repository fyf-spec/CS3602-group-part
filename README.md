# Accelerated Inference

高效 LLM 推理加速库，集成 **KV Cache 压缩** 和 **INT8 权重量化**。

## 功能特性

| 功能 | 说明 |
|------|------|
| **LazyUnifiedKVCache** | 结合 Separator + Heavy Hitter + 周期更新的 KV 压缩 |
| **H2O / LazyH2O** | Heavy Hitter Oracle 注意力驱逐 |
| **StreamingLLM** | Sink + Recent window 驱逐 |
| **INT8 Weight-Only** | 自定义 CUDA 内核的 INT8 权重量化 |

## 环境要求

### 基础要求
- Python >= 3.10
- PyTorch >= 2.0.0
- transformers == 4.33.0

### INT8 量化额外要求
- GPU: NVIDIA Ampere 及以上 (A100, RTX 30xx/40xx)，需支持 sm_80+
- CUDA: 12.x (与 PyTorch 版本对应)
- 编译器: Visual Studio Build Tools (Windows) 或 GCC (Linux)

## 安装

```bash
# 创建环境
conda create -n accel_infer python=3.10 -y
conda activate accel_infer

# 安装第三方库
pip install torch
pip install accelerate
# 基础安装 (仅 Python，无 INT8)
pip install -e .

# 完整安装 (含 INT8 CUDA 扩展)
pip install -e . --no-build-isolation
```

## quickstart

### 1. 生成 INT8 量化权重

```bash
python -m accelerated_inference.quantization.quantize \
    --model_path /home/xlyu/models/models--EleutherAI--pythia-2.8b/ \
    --out_dir checkpoints/pythia-2.8b-int8
```

### 2. PPL 评估

```bash
# FP16 Baseline
python evaluate/eval_ppl_pg19.py --mode baseline

# LazyUnified (KV 压缩)
python evaluate/eval_ppl_pg19.py --mode lazy_unified

# INT8 + LazyUnified
python evaluate/eval_ppl_pg19.py --mode int8_lazy_unified \
    --ckpt_dir checkpoints/pythia-2.8b-int8
```

### 3. 速度 Benchmark

```bash
python evaluate/eval_speed_benchmark.py --mode baseline  --model_name_or_path $Your_Local_Path 
python evaluate/eval_speed_benchmark.py --mode int8_baseline --ckpt_dir checkpoints/pythia-2.8b-int8 --model_name_or_path /home/xlyu/models/models--EleutherAI--pythia-2.8b/
python evaluate/eval_speed_benchmark.py --mode int8_lazy_unified \
    --ckpt_dir checkpoints/pythia-2.8b-int8 --model_name_or_path /home/xlyu/models/models--EleutherAI--pythia-2.8b/
```


## 目录结构

```
accelerated_inference/
├── accelerated_inference/
│   ├── kvpress/presses/          # KV Cache 压缩算法
│   │   ├── unified_press.py      # UnifiedKVCache, LazyUnifiedKVCache
│   │   └── benchmark_presses.py  # StreamingLLM, SepLLM
│   ├── quantization/             # INT8 量化
│   │   ├── int8_weight_only/     # CUDA 扩展
│   │   ├── quantize.py           # 量化脚本
│   │   └── load_int8_model.py    # 模型加载
│   └── utils.py                  # H2O, LazyH2O
├── evaluate/                     # 评估脚本
├── checkpoints/                  # 量化权重
└── outputs/                      # 结果输出
```

## License

MIT License
