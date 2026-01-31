# AMD GPU编程入门-算子优化
## 环境准备

驱动安装

==？==

查看ROCm驱动版本

```
rocm-smi
```

![image-20260126172222248](./images/20260126172232210.png)

到pytorch官网，安装对应版本的torch

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

*(注：rocm6.0 代表支持 ROCm 6.0 驱动版本，请根据 `rocm-smi` 看到的版本选择)*

## PyTorch on ROCm

ROCm (Radeon Open Compute) 之于 AMD，就像 CUDA 之于 NVIDIA。

- 它包含驱动、编译器、库（rocBLAS, rocDNN 等）。
- **关键点**：它的核心编程模型叫 **HIP**。HIP 的语法和 CUDA 几乎一模一样（`hipMalloc` vs `cudaMalloc`）。PyTorch 的底层已经通过 HIP 适配好了，上层用户无感。
- **兼容性**：PyTorch 为了兼容性，在代码层面强制使用了 `torch.cuda` 这个名字（这样原本写给 NVIDIA 的代码不用修改就能直接在 AMD 上跑）。在 ROCm 版的 PyTorch 中，`torch.cuda` 命名空间被保留并重定向到底层的 HIP 调用。
	- 打开 Python 验证一下：

```
import torch
print(f"CUDA available: {torch.cuda.is_available()}") 
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"ROCm Version: {torch.version.hip}")
```

> CUDA available: True
> Device Name: AMD Radeon 8060S
> ROCm Version: 7.1.25424-4179531dcd

## 大模型 (LLM) 推理实战

### 环境准备

**安装基础库**：

```
pip install transformers accelerate
```

**模型下载–使用 ModelScope (魔搭社区)**

```
pip install modelscope
```

在终端输入 `python` 进入交互模式：

```
from modelscope import snapshot_download

# 下载到当前目录
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='./')
print(f"模型已下载到: {model_dir}")
```

### 推理脚本

```
import torch
import time
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# 核心配置区
# ==========================================

# 模型路径 (请确保这里是你下载好的模型文件夹路径)
MODEL_PATH = "./Qwen/Qwen2.5-7B-Instruct"  

# 设备选择 (你的显卡)
DEVICE = "cuda:0" 

# ==========================================

def run_inference():
    print(f"=== AMD ROCm 推理测试 ===")

    # 打印设备信息
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"使用设备: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("[警告] 未检测到 ROCm/CUDA 设备，将使用 CPU 运行（极慢）")

    # 加载 Tokenizer
    print("\n[1/3] 正在加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"[错误] Tokenizer 加载失败: {e}")
        return


    print("\n[2/3] 正在加载模型权重 (BFloat16)...")
    st = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,  # AMD MI系列/新卡强烈推荐 BF16
            device_map=DEVICE,
            trust_remote_code=True,
            # 【关键修改】这里去掉了 attn_implementation="flash_attention_2"
            # 让 transformers 自动选择最佳实现 (通常是 sdpa)
        )
    except Exception as e:
        print(f"[致命错误] 模型加载失败: {e}")
        print("如果是显存不足，请尝试减小 batch size 或使用量化模型。")
        return
        
    print(f"模型加载耗时: {time.time() - st:.2f} 秒")

    prompt = "你好，请用这台高性能显卡为我写一首关于 AMD 显卡逆袭的七言绝句。"
    messages = [
        {"role": "system", "content": "你是一个才华横溢的诗人。"},
        {"role": "user", "content": prompt}
    ]
    

    print("\n[3/3] 开始推理...")
    
 
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    

    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    # 计时开始
    st = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,       # 最大生成长度
            temperature=0.7,          # 随机性 (0.7 比较均衡)
            top_p=0.9,
            # 设置 pad_token_id 避免之前的 Warning
            pad_token_id=tokenizer.eos_token_id 
        )
    et = time.time()

    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[:, input_len:]
    
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # 计算速度
    tokens_gen = output_ids.shape[1]
    speed = tokens_gen / (et - st)

    print("\n" + "="*20 + " 生成结果 " + "="*20)
    print(response)
    print("="*50)
    print(f"生成速度: {speed:.2f} tokens/s")
    print(f"显存占用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    run_inference()

```

>== AMD ROCm 推理测试 ==
>使用设备: AMD Radeon 8060S (62.5 GB)
>
>[1/3] 正在加载 Tokenizer...
>
>[2/3] 正在加载模型权重 (BFloat16)...
>`torch_dtype` is deprecated! Use `dtype` instead!
>Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.43it/s]
>模型加载耗时: 3.67 秒
>
>[3/3] 开始推理...
>The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
>
>== 生成结果 ==
>昔日显皇N独尊，今朝AMD起风云。
>
> Radeon力挽狂澜，性能超越旧王尊。
>
>生成速度: 9.28 tokens/s
>显存占用: 14.28 GB

### 第一步：拆解“黑盒”—— 验证 PyTorch 到底调用了谁？

**1. 看看 PyTorch 的真面目**，我们将使用 Linux 的显微镜工具 `ldd`，查看 PyTorch 的核心库到底依赖了什么文件：

```
# 找到 torch 库的路径
TORCH_LIB=$(python -c "import torch; print(torch.__file__)" | sed 's/__init__.py/lib/')

# 查看核心依赖 (过滤出 amd/rocm 相关的)
ldd $TORCH_LIB/libtorch_python.so | grep -E "amd|hip|hsa"
```

>root@aimax395:/lwh# ldd $TORCH_LIB/libtorch_python.so | grep -E "amd|hip|hsa"
>        libtorch_hip.so => /opt/venv/lib/python3.13/site-packages/torch/lib/libtorch_hip.so (0x0000758a2ff9f000)
>        libc10_hip.so => /opt/venv/lib/python3.13/site-packages/torch/lib/libc10_hip.so (0x0000758a2fe26000)
>        libamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7 (0x00007589c67f2000)
>        libhiprtc.so.7 => /opt/rocm/lib/libhiprtc.so.7 (0x00007589c60a6000)
>        libhipblas.so.3 => /opt/rocm/lib/libhipblas.so.3 (0x00007589c5fca000)
>        libhipfft.so.0 => /opt/rocm/lib/libhipfft.so.0 (0x00007589c5fb7000)
>        libhiprand.so.1 => /opt/rocm/lib/libhiprand.so.1 (0x00007589c5faf000)
>        libhipsparse.so.4 => /opt/rocm/lib/libhipsparse.so.4 (0x00007589c5f6a000)
>        libhipsolver.so.1 => /opt/rocm/lib/libhipsolver.so.1 (0x00007589c5f25000)
>        libhipsparselt.so.0 => /opt/rocm/lib/libhipsparselt.so.0 (0x00007589c5a63000)
>        libhipblaslt.so.1 => /opt/rocm/lib/libhipblaslt.so.1 (0x00007589a1c29000)
>        libamd_comgr.so.3 => /opt/rocm/lib/libamd_comgr.so.3 (0x0000758952283000)
>        libhsa-runtime64.so.1 => /opt/rocm/lib/libhsa-runtime64.so.1 (0x0000758951e66000)
>        libdrm_amdgpu.so.1 => /lib/x86_64-linux-gnu/libdrm_amdgpu.so.1 (0x0000758924586000)

请注意看这几个关键文件，它们构成了整个 **ROCm 软件栈** 的核心：

1. **`libamdhip64.so.7` (HIP Runtime)**
	- 这就是 **“翻译官”**。
	- 当写下 `torch.cuda.xxx` 时，PyTorch 其实是在调用这个库。它负责把 CUDA 风格的 API 调用，实时转换成 AMD 的指令。它是 ROCm 对应 NVIDIA `libcudart` 的核心组件。
2. **`libhsa-runtime64.so.1` (HSA Runtime)**
	- 这就是 **“工头”**。
	- HIP 只是翻译，真正负责调度 GPU、管理内存、让 GPU 也就是 Radeon 8060S 开始干活的，是底层的 HSA (Heterogeneous System Architecture) 运行时，它是 AMD 异构计算的灵魂。
3. **`libhipblas.so`, `libhipfft.so` 等**
	- 这就是 **“技能包”**。
	- 这是 AMD 已经写好的高性能数学库（矩阵乘法、快速傅里叶变换）。刚才你跑 Qwen 模型时，大量的矩阵运算就是由 `libhipblas` 完成的。
4. **`libamd_comgr.so.3` (Code Object Manager)**
	- 这就是 **“编译器前端”**。
	- 它负责动态地把代码编译成显卡能执行的二进制对象。

### 动手写“翻译官”能看懂的代码

既然我们知道中间有一层 **HIP**，那我们能不能跳过 Python，直接用 C++ 写一段代码让 HIP 跑起来？这能让我们彻底理解编译器在干什么。

```
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib> 

// 定义一个宏来检查 HIP API 的返回值
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP Error: " << hipGetErrorString(err)                \
                      << " at line " << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// 这是一个“核函数”(Kernel)，它将 Radeon 8060S 显卡上运行
// __global__ 是告诉编译器：这个函数在 GPU 上跑，但由 CPU 调用
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // 获取当前线程的 ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i]; // 每个线程只算一个数的加法
    }
}

int main() {
    int n = 1024; // 向量长度
    size_t bytes = n * sizeof(float);

    // 1. 在 CPU (Host) 上分配内存
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // 初始化数据
    for(int i=0; i<n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 2. 在 GPU (Device) 上分配显存
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // 3. 把数据从 CPU 搬运到 GPU (H2D)
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    // 4. 启动核函数！让显卡干活
    // 语法: <<<GridDim, BlockDim>>>
    // 这里开启 1 个 Block，里面有 1024 个线程并行计算
    hipLaunchKernelGGL(vector_add, dim3(1), dim3(n), 0, 0, d_a, d_b, d_c, n);
    
    // 等待 GPU 干完活
    HIP_CHECK(hipDeviceSynchronize());

    // 5. 把结果搬回 CPU (D2H)
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

    // 验证结果
    std::cout << "Element [0]: " << h_a[0] << " + " << h_b[0] << " = " << h_c[0] << std::endl;
    std::cout << "Element [1023]: " << h_a[1023] << " + " << h_b[1023] << " = " << h_c[1023] << std::endl;
    std::cout << ">>> ROCm HIP Kernel executed successfully on AMD GPU!" << std::endl;

    // 清理内存
    HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_b)); HIP_CHECK(hipFree(d_c));
    free(h_a); free(h_b); free(h_c);

    return 0;
}

```

1. 在服务器上创建一个文件 `hello_rocm.cpp`。
2. 把上面的 C++ 代码粘贴进去。
3. 使用 `hipcc` 编译并运行。

```
# 确认 hipcc 编译器是否就绪
which hipcc
# 编译并运行
hipcc hello_rocm.cpp -o hello_rocm && ./hello_rocm
```

>root@aimax395:/lwh# hipcc hello_rocm.cpp -o hello_rocm
>root@aimax395:/lwh# ./hello_rocm
>Element [0]: 1 + 2 = 3
>Element [1023]: 1 + 2 = 3
>
>ROCm HIP Kernel executed successfully on AMD GPU!

### 自己手写一个 PyTorch 算并调用

刚才我们是脱离 Python 单独跑的 C++。现在我们要**回到 Python**，利用 PyTorch 的“实时编译”(JIT) 能力，把刚才那段 C++ 代码直接嵌入到 Python 脚本里跑起来。

这正是 PyTorch 强大之处：**它允许你像写 Python 一样写 C++ 扩展。**

在当前目录新建两个文件：`my_kernel.hip` 和 `run_op.py`。

`my_kernel.hip`

```
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>

// ==========================================
// 1. GPU 核函数 (真正干活的)
// ==========================================
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// ==========================================
// 2. C++ 接口函数 (负责分配任务)
// ==========================================
torch::Tensor custom_add(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto c = torch::empty_like(a);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // 启动 GPU 核函数
    hipLaunchKernelGGL(vector_add_kernel, 
                       dim3(blocks), dim3(threads), 0, 0, 
                       a.data_ptr<float>(), 
                       b.data_ptr<float>(), 
                       c.data_ptr<float>(), 
                       size);
    return c;
}

// ==========================================
// 3. PyTorch 绑定 (PyBind11)
//    这里我们把绑定逻辑直接写在 HIP 文件里
// ==========================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add", &custom_add, "A custom vector addition kernel");
}

```

 `run_op.py`

```
import torch
from torch.utils.cpp_extension import load
import time

print("=== 开始编译自定义 HIP 算子 ===")
st = time.time()

my_module = load(
    name="my_test_op",             # 模块名字
    sources=["my_kernel.hip"],     # 你的 HIP 源码文件
    extra_cuda_cflags=["-O3"],     # 开启优化
    verbose=True                   # 打印详细编译日志
)

print(f"=== 编译完成，耗时: {time.time() - st:.2f} 秒 ===")

# ================= 测试部分 =================
print("\n[测试] 准备数据...")
size = 1024 * 1024 * 10 # 1000万个元素
x = torch.randn(size, device="cuda")
y = torch.randn(size, device="cuda")

print("[测试] 运行自定义算子...")
# 第一次运行可能会有少许冷启动开销
torch.cuda.synchronize()
t1 = time.time()
z_custom = my_module.custom_add(x, y)
torch.cuda.synchronize()
t2 = time.time()
print(f"自定义算子耗时: {(t2 - t1)*1000:.3f} ms")

print("[测试] 运行 PyTorch 原生算子...")
torch.cuda.synchronize()
t3 = time.time()
z_native = x + y
torch.cuda.synchronize()
t4 = time.time()
print(f"原生算子耗时:   {(t4 - t3)*1000:.3f} ms")

# 验证正确性
diff = (z_custom - z_native).abs().max().item()
if diff < 1e-4:
    print("\n✅ 结果正确！最大误差: ", diff)
else:
    print("\n❌ 结果错误！最大误差: ", diff)

```

>=== 开始编译自定义 HIP 算子 ===
>/lwh/my_kernel.hip -> /lwh/my_kernel.hip [skipped, no changes]
>Successfully preprocessed all matching files.
>Total number of unsupported CUDA function calls: 0
>
>Total number of replaced kernel launches: 0
>[1/2] /opt/rocm-7.1.0/bin/hipcc  -DWITH_HIP -DTORCH_EXTENSION_NAME=my_test_op -DTORCH_API_INCLUDE_EXTENSION_H -isystem /opt/venv/lib/python3.13/site-packages/torch/include -isystem /opt/venv/lib/python3.13/site-packages/torch/include/torch/csrc/api/include -isystem /opt/venv/lib/python3.13/site-packages/torch/include/THH -isystem /opt/rocm-7.1.0/include -isystem /usr/include/python3.13 -fPIC -std=c++17 -DHIP_PLATFORM_AMD=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC -DCUDA_HAS_FP16=1 -DHIP_NO_HALF_OPERATORS=1 -DHIP_NO_HALF_CONVERSIONS=1 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 --offload-arch=gfx908 --offload-arch=gfx90a --offload-arch=gfx942 --offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1200 --offload-arch=gfx1201 --offload-arch=gfx950 --offload-arch=gfx1151 --offload-arch=gfx1150 -fno-gpu-rdc -O3 -c /lwh/my_kernel.hip -o my_kernel.cuda.o
>[2/2] c++ my_kernel.cuda.o -shared -L/opt/venv/lib/python3.13/site-packages/torch/lib -lc10 -lc10_hip -ltorch_cpu -ltorch_hip -ltorch -ltorch_python -L/opt/rocm-7.1.0/lib -lamdhip64 -o my_test_op.so
>=== 编译完成，耗时: 94.24 秒 ===
>
>[测试] 准备数据...
>[测试] 运行自定义算子...
>自定义算子耗时: 3.166 ms
>[测试] 运行 PyTorch 原生算子...
>原生算子耗时:   18.847 ms
>
>✅ 结果正确！最大误差:  0.0
