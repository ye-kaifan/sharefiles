- [给 LLM 投喂物理知识](#给-llm-投喂物理知识)
  - [常用的方案](#常用的方案)
  - [可行的方案](#可行的方案)
    - [GraphRAG](#graphrag)
    - [Fine-tuning](#fine-tuning)
- [CCNU NSC3 集群 vLLM 本地部署 DeepSeek-R1](#ccnu-nsc3-集群-vllm-本地部署-deepseek-r1)
  - [vLLM 简介](#vllm-简介)
    - [核心特性](#核心特性)
    - [安装条件](#安装条件)
  - [vLLM 安装方式一：通过搭建 python 虚拟环境安装 vLLM](#vllm-安装方式一通过搭建-python-虚拟环境安装-vllm)
    - [环境准备](#环境准备)
    - [模型下载](#模型下载)
    - [**代码准备**](#代码准备)
      - [**Python脚本**](#python脚本)
      - [创建兼容 OpenAI API 接口的服务器](#创建兼容-openai-api-接口的服务器)
    - [推理速度测试](#推理速度测试)
    - [vLLM Reference](#vllm-reference)
  - [vLLM/SGLang 安装方式二：通过拉取官方的 Docker 容器安装](#vllmsglang-安装方式二通过拉取官方的-docker-容器安装)
- [sglang 双节点部署](#sglang-双节点部署)
  - [节点1 gpu039](#节点1-gpu039)
  - [节点2 gpu040](#节点2-gpu040)
  - [sglang 官方](#sglang-官方)
  - [单一节点](#单一节点)
  - [API服务部署方式](#api服务部署方式)
- [open-webui 部署](#open-webui-部署)
- [curl](#curl)


# 给 LLM 投喂物理知识

## 常用的方案
给大语言模型投喂 Lattice QCD 知识库，通常有如下三种方式：提示词（Prompt），检索增强生成（Retrieval-Augmented Generation，RAG），微调（Fine-tuning）。

三种方式的实现原理对比：
![](figs/schematic3ways.png)

三种方式的优缺点对比：
![](figs/compare3ways.png)

图表来自：
【如何给大模型喂数据？让AI更懂你～【小白科普】】 https://www.bilibili.com/video/BV1HS421R7oL/?share_source=copy_web&vd_source=4b438f829d0c01700eb6160fae7d5ea7

## 可行的方案

提示词的方式通常只支持128k上下文，显然没法胜任海量的知识库，剩下两种方式检索增强生成和微调倒是可行的。

### GraphRAG

本地安装微软开源 GraphRAG。GraphRAG优于传统的RAG，应该能够胜任对格点QCD知识的整理。

- 优点是可以直接调用deepseek R1的API，性能强劲且便宜。
- 缺点是微软的GraphRAG只支持txt和csv的文件格式。对于公式也是当作文本来处理，图片并不支持，需要预先处理，提取出图片中的文本信息。

### Fine-tuning

微调最大的困难在于数据准备，步骤如下：
1. 收集物理领域数据：收集与特定物理领域相关的高质量数据，如物理学术论文、教科书、实验报告等。确保数据涵盖目标物理领域的核心概念和术语。
2. 数据清洗和标注：对收集到的数据进行清洗，去除噪声和不相关的信息。根据需要对数据进行标注，例如标注物理公式、概念解释、实验步骤等。
3. 数据格式化：将数据转换为模型可接受的格式，如JSONL文件，包含成对的问题和回答。

准备好了用于微调的数据，还要在本地部署deepseek R1，能够本地部署的都是Qwen模型蒸馏deepseek R1得到的，性能远不如deepseek R1。


# CCNU NSC3 集群 vLLM 本地部署 DeepSeek-R1

## vLLM 简介

vLLM 是伯克利大学 LMSYS 组织开源的大语言模型高速推理框架，旨在极大地提升实时场景下的语言模型服务的吞吐量与内存使用效率。

### 核心特性

- **高效的内存管理**：通过 `PagedAttention` 算法，`vLLM` 实现了对 `KV` 缓存的高效管理，减少了内存浪费，优化了模型的运行效率。
- **高吞吐量**：`vLLM` 支持异步处理和连续批处理请求，显著提高了模型推理的吞吐量，加速了文本生成和处理速度。与 Hugging Face Transformers 相比，vLLM 的吞吐量最多可以达到其 24 倍，文本生成推理（TGI）高出 3.5 倍，并且不需要对模型结构进行任何的改变。
- **内存效率**：vLLM 实现了 KV 缓存内存几乎零浪费，解决了大语言模型推理中的内存管理瓶颈问题。
- **硬件兼容性**：vLLM 不仅支持 NVIDIA GPU，还对 AMD GPU、Intel GPU、AWS Neuron 和 Google TPU 等市面上众多硬件架构敞开怀抱。
- **易用性**：`vLLM` 与 `HuggingFace` 模型无缝集成，支持多种流行的大型语言模型，简化了模型部署和推理的过程。兼容 `OpenAI` 的 `API` 服务器。
- **分布式推理**：框架支持在多 `GPU` 环境中进行分布式推理，通过模型并行策略和高效的数据通信，提升了处理大型模型的能力。
- **多步调度**：vLLM 引入了多步调度技术，允许一次性完成多个步骤的调度和输入准备，使得 GPU 可以连续处理多个步骤而不必每个步骤都等待 CPU 指令，从而提高了 GPU 的利用率和整体吞吐量。
- **异步输出处理**：vLLM 采用异步输出处理技术，使得输出处理与模型的执行可以并行进行，进一步提高了处理效率。
- **模型微调**：vLLM 支持使用 LoRA（Low-Rank Adaptation）技术进行模型微调，这是一种用于高效微调预训练大模型的方法。
- **开源共享**：`vLLM` 由于其开源的属性，拥有活跃的社区支持，这也便于开发者贡献和改进，共同推动技术发展。


### 安装条件

- vLLM 支持在 Linux 系统上运行。
- 需要 Python 3.8-3.11 版本，CUDA 12.1。
- 需要具有计算能力 7.0 或更高的 GPU（如 V100、T4、RTX 20xx、A100、L4、H100 等）。


## vLLM 安装方式一：通过搭建 python 虚拟环境安装 vLLM

### 环境准备  

本文基础环境如下：

```
----------------
Centos 7.7
python 3.12
cuda 12.4
CUDA DRIVER 550.54.14
pytorch 2.5.1
gcc 11
g++ 11
----------------
```

```bash
# make a virtual environment first
conda create -n cuda124_vllm python=3.12 -y
module load cuda-12.4
conda activate cuda124_vllm
conda install -c conda-forge gcc=11
conda install -c conda-forge gxx=11
pip install torch
pip install vllm
```
### 模型下载

使用 modelscope 中的 snapshot_download 函数下载模型，第一个参数为模型名称，参数 cache_dir 为模型的下载路径。

新建 `model_download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件。

```python
from modelscope import snapshot_download

model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', cache_dir='/dssg/work/kfye/deepseek', revision='master')
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `cache_dir` 为你的模型下载路径哦~


### **代码准备**

#### **Python脚本**

新建 `vllm_model.py` 文件并在其中输入以下内容，粘贴代码后请及时保存文件。下面的代码有很详细的注释，如有不理解的地方，欢迎大家提 `issue`。

首先从 `vLLM` 库中导入 `LLM` 和 `SamplingParams` 类。`LLM` 类是使用 `vLLM` 引擎运行离线推理的主要类。`SamplingParams` 类指定采样过程的参数，用于控制和调整生成文本的随机性和多样性。

`vLLM` 提供了非常方便的封装，我们直接传入模型名称或模型路径即可，不必手动初始化模型和分词器。

我们可以通过这个代码示例熟悉下 ` vLLM` 引擎的使用方式。被注释的部分内容可以丰富模型的能力，但不是必要的，大家可以按需选择，自己多多动手尝试 ~

```python
# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE']='True'

def get_completion(prompts, model, tokenizer=None, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":    
    # 初始化 vLLM 推理引擎
    model='/dssg/work/kfye/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' # 指定模型路径
    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # 指定模型名称，自动下载模型
    tokenizer = None
    # 加载分词器后传入vLLM 模型，但不是必要的。
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) 
    
    text = ["Please introduce Pade approximants for me. Please reason step by step, and put your final answer within \boxed{}.<think>\n", ] # 可用 List 同时传入多个 prompt，根据 DeepSeek 官方的建议，每个 prompt 都需要以 <think>\n 结尾，如果是数学推理内容，建议包含（中英文皆可）：Please reason step by step, and put your final answer within \boxed{}.

    # messages = [
    #     {"role": "user", "content": prompt+"<think>\n"}
    # ]
    # 作为聊天模板的消息，不是必要的。
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048) # 思考需要输出更多的 Token 数，max_tokens 设为 8K，根据 DeepSeek 官方的建议，temperature应在 0.5-0.7，推荐 0.6

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if r"</think>" in generated_text:
            think_content, answer_content = generated_text.split(r"</think>")
        else:
            think_content = ""
            answer_content = generated_text
        print(f"Prompt: {prompt!r}, Think: {think_content!r}, Answer: {answer_content!r}")
```
运行代码

```bash
python vllm_model.py
```

#### 创建兼容 OpenAI API 接口的服务器

`DeepSeek-R1-Distill-Qwen` 兼容 `OpenAI API` 协议，所以我们可以直接使用 `vLLM` 创建 `OpenAI API` 服务器。`vLLM` 部署实现 `OpenAI API` 协议的服务器非常方便。默认会在 http://localhost:8000 启动服务器。服务器当前一次托管一个模型，并实现列表模型、`completions` 和 `chat completions` 端口。

- `completions`：是基本的文本生成任务，模型会在给定的提示后生成一段文本。这种类型的任务通常用于生成文章、故事、邮件等。
- `chat completions`：是面向对话的任务，模型需要理解和生成对话。这种类型的任务通常用于构建聊天机器人或者对话系统。

在创建服务器时，我们可以指定模型名称、模型路径、聊天模板等参数。

- `--host` 和 `--port` 参数指定地址。
- `--model` 参数指定模型名称。
- `--chat-template` 参数指定聊天模板。
- `--served-model-name` 指定服务模型的名称。
- `--max-model-len` 指定模型的最大长度。


```bash
conda activate cuda124_vllm
python -m vllm.entrypoints.openai.api_server --model /dssg/work/kfye/deepseek/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --served-model-name deepseekr1-qwen-1.5b --tensor-parallel-size 8 --max-model-len 2048
python -m vllm.entrypoints.openai.api_server --model /dssg/work/kfye/deepseek/deepseek-ai/DeepSeek-R1-Distill-Llama-70B --served-model-name deepseekr1-Llama-70b --tensor-parallel-size 8 --max-model-len 2048
```

- 通过 `curl` 命令查看当前的模型列表

```bash
curl http://localhost:8000/v1/models
```

- 使用 `curl` 命令测试 `OpenAI Completions API` 


```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-R1-Distill-Qwen-7B",
        "prompt": "我想问你，5的阶乘是多少？<think>\n",
        "max_tokens": 1024,
        "temperature": 0
    }'
```


- 用 `Python` 脚本请求 `OpenAI Completions API` 


```python
# vllm_openai_completions.py
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty", # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="deepseek-qwen-1.5b",
  messages=[
    {"role": "user", "content": "我想问你，5的阶乘是多少？<think>\n"}
  ]
)

print(completion.choices[0].message)
```

```shell
python vllm_openai_completions.py
```

- `request`方式

```python
# chat_vllm_api.py
import requests

headers = {
    "Content-Type": "application/json",
}

URL = "http://localhost:8000/v1/chat/completions"

prompt = {
    "model": "deepseek-qwen-1.5b",
    "messages": [
        {"role": "system", "content": "You are a large model that excels at mathematical and physical reasoning."},
        {"role": "user", "content": "Please introduce Pade approximants for me. Please reason step by step, and put your final answer within \boxed{}."}
    ],
    "temperature": 0.6,
    "top_p": 0.95
}

resp = requests.post(URL, headers=headers, json=prompt, stream=False)
rep = resp.json()
print(rep)
```


- 用 `Python` 脚本请求 `OpenAI Chat Completions API` 


```python
# vllm_openai_chat_completions.py
from openai import OpenAI
openai_api_key = "empty" # 随便填写，只是为了通过接口参数校验
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_outputs = client.chat.completions.create(
    model="deepseek-qwen-1.5b",
    messages=[
        {"role": "user", "content": "什么是深度学习？"},
    ]
)
print(chat_outputs)
```

```shell
python vllm_openai_chat_completions.py
```

### 推理速度测试

既然 `vLLM` 是一个高效的大型语言模型推理和部署服务系统，那么我们不妨就测试一下模型的回复生成速度。看看和原始的速度相比有多大的提升。这里直接使用 `vLLM` 自带的 `benchmark_throughput.py` 脚本进行测试。可以将当前文件夹 `benchmark_throughput.py` 脚本放在 `/root/autodl-tmp/` 目录下；或者也可以自行[下载最新版脚本](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py)

下面是一些 `benchmark_throughput.py` 脚本的参数说明：

- `--model` 参数指定模型路径或名称。
- `--backend` 推理后端，可以是 `vllm`、`hf` 和 `mii`。分布对应 `vLLM`、`HuggingFace` 和 `Mii` 推理后端。
- `--input-len` 输入长度
- `--output-len` 输出长度
- `--num-prompts` 生成的 prompt 数量
- `--seed` 随机种子
- `--dtype` 数据类型
- `--max-model-len` 模型最大长度
- `--hf_max_batch_size` `transformers` 库的最大批处理大小（仅仅对于 `hf` 推理后端有效且为必填字段）
- `--dataset` 数据集路径。（如未设置会自动生成数据）



测试 `vLLM` 推理速度的命令和参数设置

```bash
python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/DeepSeek-R1-Distill-Qwen-7B \
	--backend vllm \
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2025 \
  --dtype float16 \
  --max-model-len 512
```

### vLLM Reference

- https://github.com/datawhalechina/self-llm/blob/master/models/DeepSeek-R1-Distill-Qwen/
- https://docs.vllm.ai/en/latest/api/offline_inference/llm.html


## vLLM/SGLang 安装方式二：通过拉取官方的 Docker 容器安装

在GPU上运行的话需要在GPU节点安装 docker 和 nvidia-container-toolkit，安装需要管理员权限。

由于`DockerHub`是国外网站，直接运行下面的命令会报错，
```bash
docker pull vllm/vllm-openai
```
换成国内镜像源才能顺利下载，
```bash
docker pull docker.1ms.run/vllm/vllm-openai
docker pull docker.1ms.run/lmsysorg/sglang:latest
docker pull docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124
docker pull docker.1ms.run/dyrnq/open-webui
docker pull docker.1ms.run/savatar101/marker-api
docker pull docker.1ms.run/anetaco/marker:v108
docker pull registry.cn-beijing.aliyuncs.com/anetaco/marker:v108
docker pull registry.cn-beijing.aliyuncs.com/quincyqiang/mineru:0.1-models
docker pull registry.cn-beijing.aliyuncs.com/savatar101/marker-api
```
> 镜像源的网址经常换，需要自己搜索`DockerHub 国内镜像源`寻找可用的镜像源网址，用法类似。

ln02节点
```bash
docker save -o /dssg/work/kfye/docker/images_tar/vllm0.7.2.tar docker.1ms.run/vllm/vllm-openai
docker save -o /dssg/work/kfye/docker/images_tar/sglang0.4.2.post4-cu124.tar docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124
```
gpu039节点
```bash
docker load -i /dssg/work/kfye/docker/images_tar/vllm0.7.2.tar
docker load -i /dssg/work/kfye/docker/images_tar/sglang0.4.2.post4-cu124.tar
```


# sglang 双节点部署

程序员阿赞
## 节点1 gpu039
```bash
docker run --gpus all \
    --shm-size 10g \
    --network=host \
    -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --name sglang2a \
    -p 1234:1234 -d \
    --ipc=host --restart always \
    docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1 --served-model-name deepseek-r1 --tp 16 --dist-init-addr 11.11.11.39:4321 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 1234
```

## 节点2 gpu040
```bash
docker run --gpus all \
    --shm-size 10g \
    --network=host \
    -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --name sglang2a \
    -p 1234:1234 -d \
    --ipc=host --restart always \
    docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1 --served-model-name deepseek-r1 --tp 16 --dist-init-addr 11.11.11.39:4321 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 1234
```

optional
```bash
--privileged -e NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 -e NCCL_P2P_LEVEL=NVL -e NCCL_IB_GID_INDEX=0  -e NCCL_IB_CUDA_SUPPORT=1 -e library/NCCL_IB_DISABLE=0 -e NCCL_SOCKET_IFNAME=ibs11,ibs12,ibs13,ibs14,ibs15,ibs16,ibs17,ibs18 -e NCCL_DEBUG=INFO -e NCCL_NET_GDR_LEVEL=2 \
--privileged -e NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 -e NCCL_P2P_LEVEL=NVL -e NCCL_IB_GID_INDEX=0  -e NCCL_IB_CUDA_SUPPORT=1 -e NCCL_IB_DISABLE=0 -e NCCL_SOCKET_IFNAME=ibs11,ibs12,ibs13,ibs14,ibs15,ibs16,ibs17,ibs18 -e NCCL_DEBUG=INFO -e NCCL_NET_GDR_LEVEL=2 \
```

## sglang 官方
```bash
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --name sglang1 \
    -it \
    --rm \
    --ipc=host \
    docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1 --tp 16 --dist-init-addr 10.10.10.39:20000 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 40000 --output-file "deepseekr1.jsonl"

```

## 单一节点
```bash
# gpu039
docker run --gpus all --shm-size 10g --name sglang1b \
    -p 22:22 -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --ipc=host docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 --trust-remote-code --host 11.11.11.39 --port 22
docker: Error response from daemon: driver failed programming external connectivity on endpoint sglang1b (265c3c57cc5f76d4304a66097f4e000a415bb9f7fcf440123246a7766ca1f0dd): failed to bind port 0.0.0.0:22/tcp: Error starting userland proxy: listen tcp4 0.0.0.0:22: bind: address already in use.

docker run --gpus all --shm-size 10g --name sglang1a \
    -p 1234:1234 -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --ipc=host docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 --trust-remote-code --host 11.11.11.39 --port 1234
[2025-02-17 11:07:35] ERROR:    [Errno 99] error while attempting to bind on address ('11.11.11.39', 1234): cannot assign requested address

docker run --gpus all --shm-size 10g --name sglang1a \
    -p 22:4321 -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --ipc=host docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 --trust-remote-code --host 11.11.11.39 --port 4321

docker run --gpus all --shm-size 10g --name sglang1d \
    -p 1234:1234 -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --network host --ipc=host docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 --trust-remote-code --host 11.11.11.39 --port 1234
已经在gpu039节点成功部署。
部署qwen-32b
docker cp /dssg/work/kfye/qwen/qwen32b sglang_gpu4:/sgl-workspace/
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server --model-path /sgl-workspace/qwen32b --served-model-name qwen-32b --tp 2 --trust-remote-code --host 0.0.0.0 --port 1234


docker run --gpus all --shm-size 32g --name sglang_gpu4 -itd \
    -v /dssg/work/kfye/deepseek/deepseek-ai/DeepSeek-R1-Distill-Llama-70B:/sgl-workspace/DeepSeek-R1-Distill-Llama-70B \
    -v /dssg/work/kfye/deepseek/deepseek-ai/pychat:/sgl-workspace/pychat \
    --network host --ipc=host docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124

# gpu040
docker run --gpus all --shm-size 10g --name sglang1b \
    -p 1234:1234 -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --ipc=host docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 --trust-remote-code --host 11.11.11.40 --port 1234
[2025-02-17 10:22:33] ERROR:    [Errno 99] error while attempting to bind on address ('11.11.11.40', 1234): cannot assign requested address

docker run --gpus all --shm-size 10g --name sglang1c \
    -p 1234:1234 -v /dssg/work/kfye/deepseek/deepseek-ai:/sgl-workspace/deepseek-ai \
    --network host --ipc=host docker.1ms.run/lmsysorg/sglang:v0.4.2.post4-cu124 \
    python3 -m sglang.launch_server --model-path /sgl-workspace/deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 --trust-remote-code --host 11.11.11.40 --port 1234
```

## API服务部署方式
```bash
在gpu039节点:
docker ps -a
docker start sglang1a
nvidia-smi # 直到gpu显存利用率达到90%左右就是api服务部署完成了
docker logs sglang1a # 报错不用管，是上次 docker stop sglang1a 带来的，等一会就会出现加载模型的信息，需要十几分钟
docker cp /dssg/work/kfye/deepseek/openai_chat_completion_streaming.py sglang2:/sgl-workspace/
docker exec -it sglang1a bash
python3 openai_chat_completion_streaming.py
```


# open-webui 部署

由于`DockerHub`是国外网站，直接运行下面的命令会报错，
```bash
docker pull vllm/vllm-openai
```
换成国内镜像源才能顺利下载，
```bash
docker pull docker.1ms.run/dyrnq/open-webui
```
> 镜像源的网址经常换，需要自己搜索`DockerHub 国内镜像源`寻找可用的镜像源网址，用法类似。

```bash
docker save -o /dssg/work/kfye/docker/images_tar/openwebui.tar docker.1ms.run/dyrnq/open-webui
docker load -i /dssg/work/kfye/docker/images_tar/openwebui.tar
docker run -d -p 1234:8080 -v open-webui:/app/backend/data --name open-webui docker.1ms.run/dyrnq/open-webui
```

# curl

```bash
curl http://122.204.190.7:5678/v1/chat/completions \
curl http://122.204.190.7:8555/v1/chat/completions \
curl http://122.204.190.7:1234/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen-32b",
    "messages": [
        {"role": "user", "content": "Why the sky is blue?\n"}
    ]
}'


"model": "minicpm-v:8b-2.6-fp16",
"model": "DeepSeek-R1-Distill-Llama-70B",
"model": "deepseek-r1-70b:latest",


curl http://11.11.11.40:11434/v1/chat/completions \
curl http://11.11.11.39:1234/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen-32b",
    "messages": [
        {"role": "user", "content": "Why the sky is blue?<think>\n"}
    ]
}'

curl http://127.0.0.1:11434/api/chat -d '{
curl http://localhost:11434/api/chat -d '{
curl http://11.11.11.40:11434/api/chat -d '{
curl http://122.204.190.7:11434/api/chat -d '{
  "model": "qwen2.5:7b-instruct-120k",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ],
  "stream": false
}'


curl --request POST \
     --url http://122.204.190.7:8380/api/v1/chats/ad54d9d2f02e11ef90fa0242ac120006/sessions \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer ragflow-JlMjRkOWYyZjAzMjExZWZiOTBiMDI0Mm' \
     --data '{
    "name":"new_chat_1"
}'

curl --request POST \
     --url http://122.204.190.7:8380/api/v1/chats/ad54d9d2f02e11ef90fa0242ac120006/sessions \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer ragflow-JlMjRkOWYyZjAzMjExZWZiOTBiMDI0Mm' \
     --data-binary '
     {
          "name":"new_chat_1"
          "question": "Who are you",
          "stream": true,
     }'

"session_id":"9fa7691cb85c11ef9c5f0242ac120005"
"dataset_ids": ["0b2cbc8c877f11ef89070242ac120005"],

ad54d9d2f02e11ef90fa0242ac120006
```
