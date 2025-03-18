- [marker-pdf 安装与使用指南](#marker-pdf-安装与使用指南)
  - [环境配置与安装](#环境配置与安装)
  - [多 GPU 环境下的 PDF 处理](#多-gpu-环境下的-pdf-处理)
    - [配置文件修改](#配置文件修改)
    - [marker\_chunk\_convert 命令详解](#marker_chunk_convert-命令详解)
  - [调用 ollama 部署的 LLM 优化OCR结果](#调用-ollama-部署的-llm-优化ocr结果)
    - [Ollama 服务部署](#ollama-服务部署)
    - [模型吞吐性能评测](#模型吞吐性能评测)
    - [永久修改ollama模型参数：创建新模型](#永久修改ollama模型参数创建新模型)
    - [ollama用本地模型权重创建新模型](#ollama用本地模型权重创建新模型)
    - [PDF 处理示例](#pdf-处理示例)

# marker-pdf 安装与使用指南

## 环境配置与安装

首先，创建一个新的 conda 环境并安装必要的依赖：

```bash
conda create -n marker python=3.12 -y
conda activate marker
module load cuda-12.4
pip install torch
pip install marker-pdf
```

安装完成后，可以验证安装路径：

```bash
(marker) [kfye@gpu040 kfye]$ which marker
/dssg/work/kfye/anaconda3/envs/marker/bin/marker
(marker) [kfye@gpu040 kfye]$ which marker_single
/dssg/work/kfye/anaconda3/envs/marker/bin/marker_single
(marker) [kfye@gpu040 kfye]$ which marker_chunk_convert
/dssg/work/kfye/anaconda3/envs/marker/bin/marker_chunk_convert
(marker) [kfye@gpu040 kfye]$ which conda
/dssg/work/kfye/anaconda3/condabin/conda
```

## 多 GPU 环境下的 PDF 处理

### 配置文件修改

首先需要修改转换脚本的配置：

```bash
vim /dssg/work/kfye/anaconda3/envs/marker/lib/python3.12/site-packages/marker/scripts/chunk_convert.sh
```

关键配置内容：
```bash
cmd="CUDA_VISIBLE_DEVICES=$DEVICE_NUM marker $INPUT_FOLDER --output_dir $OUTPUT_FOLDER --num_chunks $NUM_DEVICES --chunk_idx $DEVICE_NUM --workers $NUM_WORKERS --output_format markdown --paginate_output --force_ocr --languages \"en,zh\""
# cmd="CUDA_VISIBLE_DEVICES=$DEVICE_NUM marker $INPUT_FOLDER --output_dir $OUTPUT_FOLDER --num_chunks $NUM_DEVICES --chunk_idx $DEVICE_NUM --workers $NUM_WORKERS"
```

### marker_chunk_convert 命令详解

```bash
# 使用 4 张GPU卡，每张卡 3 个工作进程
time NUM_DEVICES=4 NUM_WORKERS=3 marker_chunk_convert /dssg/work/kfye/marker/pdfs /dssg/work/kfye/marker/mds > /dssg/work/kfye/marker/log_marker.txt 2>&1 &
tail -f /dssg/work/kfye/marker/log_marker.txt
```

参数详解：
- `NUM_DEVICES=2`代表调用前两张GPU卡。默认`CUDA_VISIBLE_DEVICES`从0开始分配，命令前面加`CUDA_VISIBLE_DEVICES=3`也没用，只能自己改`marker/scripts/chunk_convert.sh`。
- 一个pdf只能在一张GPU卡上跑，目录下有多个PDF的话，marker会自动分到四个GPU（比如17个PDF自动分成5+5+5+2在4张卡跑，自动分成9+8在2张卡跑）。
- `marker_chunk_convert`前面添加`NUM_WORKERS=3`指的是每张GPU卡3个进程。
- `NUM_WORKERS=1`代表单一进程，gpu利用率大部份时间都在15%-20%，偶尔能到50%。单进程显存占用 2523MB-3348MB
- `NUM_WORKERS`设置成大于等于4，可以把GPU利用率拉满，但是会报错`libgomp: Thread creation failed: Resource temporarily unavailable`

想中断`marker_chunk_convert`进程的话，`Ctrl+C`不管用，只能`pgrep -f marker | xargs kill -9`，建议先用`pgrep -fl marker`看一下进程的完整名称，别误杀进程了。

## 调用 ollama 部署的 LLM 优化OCR结果

### Ollama 服务部署

```bash
CUDA_VISIBLE_DEVICES=6,7 OLLAMA_NUM_PARALLEL=12 \
OLLAMA_KEEP_ALIVE=24h OLLAMA_HOST=0.0.0.0:11434 \
ollama serve > /dssg/work/kfye/marker/log_ollama_serve.txt 2>&1 &
```

### 模型吞吐性能评测

PDF转md的速度受限于Ollama部署的LLM的吞吐速度。不同模型的推理速度测试结果：

```bash
# 各模型的评测结果
ollama run deepseek-r1:32b-qwen-distill-fp16 --verbose # eval rate: 21.51 tokens/s
ollama run qwen2.5:72b-instruct --verbose # eval rate: 22.00 tokens/s
ollama run qwen2.5:1.5b # eval rate: 177.93 tokens/s
ollama run qwen2.5:7b-instruct # eval rate: 109.82 tokens/s
ollama create deepseek-r1-q4km # fail
# msg="model missing blk.0 layer size"
# panic: interface conversion: interface {} is nil, not *llm.array
```

### 永久修改ollama模型参数：创建新模型
```bash
CUDA_VISIBLE_DEVICES=1,2 OLLAMA_NUM_PARALLEL=2 \
OLLAMA_KEEP_ALIVE=24h OLLAMA_HOST=0.0.0.0:11434 \
ollama serve > /dssg/work/kfye/marker/log_ollama_serve.txt 2>&1 &

ollama show --modelfile qwen2.5:7b-instruct > Modelfile
vim Modelfile
......
FROM qwen2.5:7b-instruct
PARAMETER temperature 0.6
PARAMETER num_ctx 120000
......
ollama create qwen2.5:7b-instruct-120k -f Modelfile
```

### ollama用本地模型权重创建新模型

```bash
# deepseek-r1-70b, f16精度
cd /path/to/safetensor_or_gguf_files
vim Modelfile
---
FROM .

PARAMETER num_predict 500
PARAMETER temperature 0.6
PARAMETER num_batch 128
PARAMETER num_ctx 127000
---
ollama create deepseek-r1-70b
ollama create deepseek-r1-q4km # fail
# msg="model missing blk.0 layer size"
# panic: interface conversion: interface {} is nil, not *llm.array

```

### PDF 处理示例

使用 LLM 处理单个 PDF 文件：

```bash
CUDA_VISIBLE_DEVICES=5 marker_single /dssg/work/kfye/marker/pdf/LQCD_liuchuan.pdf \
--output_dir /dssg/work/kfye/marker/md_llm/qwen7b \
--output_format markdown --paginate_output --force_ocr --languages "en,zh" \
--use_llm --ollama_base_url http://0.0.0.0:11434 \
--ollama_model qwen2.5:7b-instruct-120k \
--llm_service=marker.services.ollama.OllamaService > /dssg/work/kfye/marker/log_qwen7b_liuchuan.txt 2>&1 &
```

性能说明：
- Ollama 服务使用两块 GPU，设置 OLLAMA_NUM_PARALLEL=4，GPU 利用率维持在 50% 以上，可以通过调大 OLLAMA_NUM_PARALLEL 提高利用率。

后台进程信息：
```bash
/dssg/work/kfye/deepseek/ollama/lib/ollama/runners/cuda_v12_avx/ollama_llama_server runner --model /dssg/work/kfye/.ollama/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730 --ctx-size 4096 --batch-size 512 --n-gpu-layers 29 --threads 128 --parallel 2 --port 33959
```
