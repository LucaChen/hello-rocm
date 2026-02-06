# 第 1 章 实战 2：Qwen 2.5 模型推理 Demo（vLLM + ROCm）

### 1.2.4 实战 2：Qwen 2.5 模型推理 Demo（vLLM + ROCm）

> 🚀 **实战目标**：本节展示如何在 AMD GPU 上通过 **vLLM + ROCm 7** 运行阿里 Qwen2.5 系列大模型的推理。
>
> 💡 **适用提示**：本示例以 Qwen2.5-7B-Instruct 为例，适合桌面 Radeon 和数据中心 Instinct 系列 GPU。

#### 步骤 1：使用 Docker 启动 vLLM 环境

使用 Docker 可以快速获得一个预配置好的 vLLM + ROCm 环境：

```bash
docker run -it \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 8G \
  -v $(pwd):/workspace \
  --name vllm \
  rocm/vllm-dev:rocm7.2_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.14.0rc0
```

**参数说明**：

| 参数 | 说明 |
| :--- | :--- |
| `--network=host` | 使用主机网络，便于访问服务 |
| `--device=/dev/kfd --device=/dev/dri` | 挂载 GPU 设备 |
| `--group-add=video` | 添加到 video 组以访问 GPU |
| `--ipc=host --shm-size 8G` | 共享内存配置，用于多进程通信 |
| `-v $(pwd):/workspace` | 挂载当前目录到容器的 /workspace |

#### 步骤 2：环境准备

进入容器后，安装基础库：

```bash
pip install transformers accelerate
```

#### 步骤 3：下载模型（使用 ModelScope）

安装 ModelScope：

```bash
pip install modelscope
```

在终端输入 `python` 进入交互模式：

```python
from modelscope import snapshot_download

# 下载到当前目录
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./')
print(f"模型已下载到: {model_dir}")
```

**输出示例：**

```text
模型已下载到: ./Qwen/Qwen2___5-7B-Instructors
```

#### 步骤 4：启动 vLLM 推理服务

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./Qwen/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 \
  --port 3000 \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --swap-space 16 \
  --disable-log-requests \
  --tensor-parallel-size 1 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --max-model-len 32768 \
  --distributed-executor-backend mp
```

**参数说明**：

| 参数 | 说明 |
| :--- | :--- |
| `--model` | 模型路径 |
| `--dtype float16` | 使用半精度浮点数 |
| `--gpu-memory-utilization 0.9` | GPU 显存利用率 |
| `--swap-space 16` | Swap 空间大小（GB） |
| `--max-model-len 32768` | 最大上下文长度 |

#### 步骤 5：测试推理服务

使用 curl 发送请求：

```bash
curl -s http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Qwen/Qwen2___5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "用一句话介绍一下 Qwen2.5-7B-Instruct。"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }' | jq .
```

#### ✅ 预期结果

如果一切正常，你会收到类似以下的 JSON 响应，包含 Qwen2.5 模型生成的回答：

![alt text](images/Qwen2.5_vllm.png)

---


