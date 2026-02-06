# 第 1 章 实战 3：Qwen 2.5 原生 PyTorch 推理

### 1.2.5 实战 3：Qwen 2.5 原生 PyTorch 推理

> 🎯 **实战目标**：本节展示如何不依赖 vLLM 等推理框架，直接使用 **PyTorch + Transformers** 在 AMD GPU 上运行 Qwen2.5 模型推理。
>
> 💡 **适用场景**：需要更灵活的控制、研究模型内部行为、或只需简单单卡推理的场景。

#### 步骤 1：环境准备

确保已安装必要的依赖：

```bash
pip install torch transformers accelerate
```

#### 步骤 2：创建推理脚本

创建文件 `qwen_pytorch_inference.py`：

```python
# file: qwen_pytorch_inference.py
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 核心配置区
# ==========================================

# 模型路径
MODEL_PATH = "./Qwen/Qwen2___5-7B-Instruct"

# 设备选择
DEVICE = "cuda:0"

# ==========================================

def run_inference():
    print(f"=== AMD ROCm PyTorch 推理测试 ===")

    # 打印设备信息
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"使用设备: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("[警告] 未检测到 ROCm/CUDA 设备，将使用 CPU 运行（极慢）")

    # 加载 Tokenizer
    print("\n[1/3] 正在加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True,trust_remote_code=True)
    except Exception as e:
        print(f"[错误] Tokenizer 加载失败: {e}")
        return

    print("\n[2/3] 正在加载模型权重 (BFloat16)...")
    st = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,  # AMD MI系列/新卡推荐 BF16
            device_map=DEVICE,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[致命错误] 模型加载失败: {e}")
        print("如果是显存不足，请尝试使用量化模型。")
        return

    print(f"模型加载耗时: {time.time() - st:.2f} 秒")

    # 构建对话
    prompt = "你好，请用这台高性能显卡为我写一首关于 AMD 显卡逆袭的七言绝句。"
    messages = [
        {"role": "system", "content": "你是一个才华横溢的诗人。"},
        {"role": "user", "content": prompt}
    ]

    print("\n[3/3] 开始推理...")

    # 应用聊天模板
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    # 生成文本
    st = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    et = time.time()

    # 解码输出
    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[:, input_len:]

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 计算性能指标
    tokens_gen = output_ids.shape[1]
    speed = tokens_gen / (et - st)

    print("\n" + "="*20 + " 生成结果 " + "="*20)
    print(response)
    print("="*50)
    print(f"生成速度: {speed:.2f} tokens/s")
    print(f"显存占用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    # 启用实验性 ROCm 优化
    import os
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    run_inference()
```

#### 步骤 3：运行推理

```bash
python qwen_pytorch_inference.py
```

#### ✅ 预期输出

![alt text](images/Qwen2.5_torch.png)

---

## 📖 参考文献

| # | 描述 | 链接 |
| :--- | :--- | :--- |
| [1] | AMD ROCm 7.2 正式发布:支持多款新硬件,优化 Instinct AI 性能 | [链接](https://so.html5.qq.com/page/real/search_news?docid=70000021_7796976caaa35752) |
| [2] | AMD Expands AI Leadership Across Client, Graphics, and ... | [链接](https://www.amd.com/en/newsroom/press-releases/2026-1-5-amd-expands-ai-leadership-across-client-graphics-.html) |
| [3] | AI Acceleration with AMD Radeon™ Graphics Cards | [链接](https://www.amd.com/en/products/graphics/radeon-ai.html) |
| [4] | AMD ROCm 7.2 更新相关报道（IT之家等综合） | [链接](https://so.html5.qq.com/page/real/search_news?docid=70000021_9816977467427752) |
| [5] | Day 0 Support for Qwen3-Coder-Next on AMD Instinct GPUs | [链接](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen3-coder-next-on-amd-instinct-gpus.html) |
| [6] | ROCm 7 软件 | [链接](https://www.amd.com/zh-cn/products/software/rocm/whats-new.html) |
| [7] | Ubuntu 将原生支持 AMD ROCm 软件 | [链接](https://so.html5.qq.com/page/real/search_news?docid=70000021_494693a705e92252) |
| [8] | Install PyTorch via PIP (Linux ROCm) | [链接](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html) |
| [9] | Install PyTorch via PIP (Windows ROCm) | [链接](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html) |
| [10] | ResNet for image classification using AMD GPUs | [链接](https://rocm.blogs.amd.com/artificial-intelligence/resnet/README.html) |

---


