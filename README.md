# Qwen3.8-27B · EXL3 · DFlash2（Windows 部署版）

<p align="center">
  <sub>fork 自 <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a> 的部署套件 · 新增 Windows 支持</sub>
</p>

> **本 fork 说明**：上游仓库面向 Linux / DGX Spark。本 fork（`CaiJianling/Qwen3.8-27B-DFlash2-EXL3-5.0bpw-Windows`）在此基础上新增了 **Windows 启动脚本 `start.ps1`** 和配套的 **`requirements.txt`**（使用 exllamav3 官方 GitHub Release 预编译 wheel，免本地编译），并整理出本文档。原英文版 README 见 `README.md.bak`。

## 项目简介

部署套件，用于以 **OpenAI 兼容接口** 服务 **Qwen3.8-27B** 的 EXL3 3.5bpw 量化版本，支持投机解码（Speculative Decoding）：

- **MTP**（默认）—— 草稿头位于 checkpoint 内部，无需额外下载，上下文/显存性价比最高
- **DFlash2** —— 专用的 EXL3 5.0bpw 草稿模型（约 1.4 GB），约快 15%
- 附带 ~4.5-bit KV cache 通道（Ada/Hopper/Blackwell 上为 NVFP4，Ampere 上为 Hadamard-4），最大化每 GB 显存的上下文长度

模型权重托管在 Hugging Face（见[模型卡](#模型卡)），推理引擎为 [exllamav3 fork](https://github.com/MiaAI-Lab/exllamav3)（DFlash2/MTP 草稿、NVFP4/FP8 KV、aarch64 GB10 + x86 CUDA）。

## 仓库内容

| 文件 | 说明 |
|---|---|
| `start.ps1` | **Windows 启动脚本（本 fork 新增）**，环境驱动的 OpenAI 兼容服务器启动器 |
| `start.sh` | Linux/macOS 启动脚本（上游原版） |
| `stop.sh` | 停止服务器（优雅关闭，可安全重复运行） |
| `tools/serve_openai.py` | 服务器本体（chat completions、流式、**工具调用**） |
| `.env.example` | 配置模板（含注释说明） |
| `requirements.txt` | Windows 依赖清单（exllamav3 预编译 wheel + 服务依赖，**本 fork 新增**） |
| `model-cards/` | 两个 HF 权重集的模型卡 |

## 环境要求（Windows）

- Windows 10/11，NVIDIA GPU（推荐 24 GB 显存起步，如 RTX 3090 / 4090）
- Python 3.10 – 3.13（撰写时验证于 Python 3.13）
- CUDA 驱动（使用 cu128 wheel，即 CUDA 12.8 系；torch 2.10.0+cu128）
- **Visual Studio**（勾选「使用 C++ 的桌面开发」工作负载）—— 需要 `cl.exe` 编译器

## Windows 快速开始

### 1. 克隆仓库（GitHub 加速）

```powershell
git clone https://gh-proxy.org/https://github.com/CaiJianling/Qwen3.8-27B-DFlash2-EXL3-5.0bpw-Windows.git
```

### 2. 安装 Visual Studio

1. 安装 Visual Studio Installer → 安装 **Community 20xx** 版本
2. 勾选 **「使用 C++ 的桌面开发」** 工作负载（需要用到 C++ 编译）
3. 打开 **x64 Native Tools Command Prompt for VS 20xx**，确认能调用编译器：

   ```powershell
   where cl
   ```

4. 将 `cl.exe` 所在目录加入 `PATH`，确保终端可调用

### 3. 下载模型（Hugging Face 加速）

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
hf download Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw `
 --local-dir ./models/Qwen3.8-27B-EXL3-3.5bpw `
 --max-workers 6
```

> **注意**：Windows 版 **不会自动下载模型**——需将模型放到 `MODEL_DIR` 指定位置（缺省 `models\Qwen3.8-27B-EXL3-3.5bpw`，相对或绝对路径均可，可在 `.env` 中修改）。若使用 `DRAFT=dflash2`，还需把草稿模型 `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` 下载到 `DRAFT_DIR`（缺省 `models\Qwen3.8-27B-DFlash2-EXL3-5.0bpw`）。

### 4. 配置并启动

```powershell
Copy-Item .env.example .env          # 按需修改：上下文、显存、端口等
powershell -ExecutionPolicy Bypass -File start.ps1
```

- **首次运行**：自动创建 `.venv` → 安装 CUDA 版 torch 2.10（cu128 索引）→ 安装 `requirements.txt`（exllamav3 预编译 wheel ~236 MB + 服务依赖，**免编译**）
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- **之后每次**：直接启动服务器
- 就绪后自测：`curl http://localhost:8888/health`

### start.ps1 特性

- **自动引导**：首次运行创建 `.venv` 并安装依赖；缺依赖时自动补齐，之后直接启动
- **免编译**：exllamav3 通过 GitHub Release 预编译 wheel 安装（`--find-links` 走 gh-proxy 镜像加速），无需本地编译 CUDA kernel
- **自动打补丁**：修复 exllamav3 1.4.5 在 Windows 无 triton 时的两个 bug（`dsa_triton.py` 顶层 import 崩溃；`torch.py` SDPA fallback 的 `dim ≥ 512` 限制），幂等、重装引擎后自动重打
- **自动检测显存**：启动时枚举所有可见 GPU（索引/名称/显存，尊重 `CUDA_VISIBLE_DEVICES`）并打印清单；`GPU_MEM_GB` 未设置时按第一张可见卡的 `VRAM/1024 − 2`（最低 8 GB）自动推断
- **多卡推理**：`GPU_SPLIT=auto` 走 exllamav3 原生 autosplit（按各卡实时空闲显存自动分配），或 `GPU_SPLIT=13,13` 显式指定每卡预算；`TENSOR_PARALLEL=1` 开启张量并行（详见[下方多卡章节](#多卡推理双卡--多卡)）
- **排错开关**：`powershell -ExecutionPolicy Bypass -File start.ps1 -SetupOnly` 只装环境、不启动服务器
- **依赖约束**：torch 必须为 `torch==2.10.0+cu128` 且从 cu128 索引安装（PyPI 直装是 CPU 版，`torch.cuda` 不可用；且装 exllamav3 时 PyPI 会把 torch 静默替换成 CPU 版）
- **Triton 缓存**：`TRITON_CACHE_DIR` 指向仓库内 `.triton-cache`，避免 `%USERPROFILE%\.triton` 的权限问题

## Linux / macOS 快速开始（上游原版）

```bash
cp .env.example .env      # 编辑：context、显存
./start.sh                # 首次：创建 .venv + 安装引擎 + 自动下载权重，服务 http://localhost:8888/v1
```

`start.sh` 自举式启动：首次运行创建 `.venv`、安装 GPU torch 并从本 fork 源码编译安装 exllamav3 引擎（DFlash2/MTP 草稿、NVFP4/FP8 KV、aarch64 GB10 移植均为 fork 特性），并**自动从 Hugging Face 下载目标/草稿权重**（支持断点续传）。`EXL3_REPO`（`.env` 中）可覆盖引擎来源（任意 git URL 或本地路径）；`HF_TARGET_REPO` / `HF_DRAFT_REPO` 可指向镜像仓库。

## 配置说明（.env）

- `DRAFT` —— 投机解码方式：`mtp`（默认；草稿头在目标 checkpoint 内，无需下载，小显存下上下文多约 50%）、`dflash2`（专用草稿模型，token/s 最快）或 `none`。`MODEL_DIR` / `DRAFT_DIR` 分别指定目标与（dflash2 的）草稿路径
- `CONTEXT_SIZE` —— KV cache 大小（token）。原生上限 262,144；设为更大值会自动切换 YaRN 配置变体（1M 可用）
- `CACHE_QUANT` —— KV 格式：`none`（fp16）/ `8` / `8,4` / `4` / `fp8` / `nvfp4`。`nvfp4` 与 `fp8` 需要 compute capability ≥ 8.9（Ada 4090、Hopper、Blackwell / GB10）；Ampere（3090, sm_86）无法编译这些 Triton kernel，请用 Hadamard `4`（~4.5 bits/elem）或 `8` / `8,4`
  > **Windows 差异**：上游 exllamav3 的 `-cq` 只接受整数位宽（`8` = fp8 KV，`8,4` = k8/v4）；`nvfp4` 是 fork 专有格式，**上游 wheel 不支持**。量化 cache 走 triton 后端，需要 `triton-windows`（已列入 `requirements.txt`）。start.ps1 默认 `CACHE_QUANT=8`。
- `GPU_MEM_GB` —— 单卡模式（默认）下权重 + cache 的显存预算；未设置时按第一张可见 GPU 自动检测（VRAM − 2 GB，最低 8 GB）
- `GPU_SPLIT` —— 多卡配置：`auto`（exllamav3 原生 autosplit，按各卡实时空闲显存自动分配，推荐）或逗号分隔的每卡预算（如 `13,13`、`24,18.5`；值数 = 卡数）。**注意单值 `13` 按上游语义只使用 GPU0**，多卡必须写 `auto` 或 ≥2 个值
- `TENSOR_PARALLEL` —— `1/true` 开启张量并行（默认层切分）；未配 `GPU_SPLIT` 时自动等同 `auto`。与 MoE CPU offload 互斥（本启动器未启用）
- `TP_BACKEND` —— 张量并行后端：`native`（默认）/ `nccl`，仅张量并行时有效
- 多卡选卡：设置环境变量 `CUDA_VISIBLE_DEVICES=0,1` 限制参与推理的物理卡（启动器的 GPU 清单与引擎均尊重该变量）
- `CPU_CACHE_GB` —— CPU 二级缓存（已接受但暂未生效，保持 0）
- `PORT` / `HOST` —— 服务端口（默认 8888）与绑定地址（`0.0.0.0` 为局域网可访问、无鉴权）
- Windows 特有：`TORCH_INDEX_URL`（默认 `https://download.pytorch.org/whl/cu128`）/ `TORCH_SPEC`（默认 `torch==2.10.0+cu128`）

> **并发说明**：服务器一次只生成一个请求（batch-1 投机解码），并发请求自动排队。DGX Spark 上 `DRAFT=dflash2` 实测：8 个并发请求完全顺序执行（无批处理收益），聚合吞吐约 **16.7 tok/s**——如果你的负载是并发而非单请求，请按此规划容量。
>
> **推理说明**：服务器**始终会推理**，目前没有关闭它的选项；`chat_template_kwargs.enable_thinking`（vLLM/SGLang 惯例）会被静默忽略。推理轨迹通过独立的 `reasoning_content` 字段返回（完整响应和每个流式 `delta` 中都有），但**不是**独立的 token 预算——推理与可见的 `content` 共享同一个 `max_tokens`。预算过紧时（例如对短答 prompt 设 `max_tokens: 16`），可能返回完全没有 `content` 的响应（推理轨迹把预算全占掉了）。期望快速、廉价的 warmup/probe 调用时请给足预算，别以为短 `max_tokens` 就等价于短等待。

## 24 GB 显卡（RTX 3090 / 4090）

套件默认面向 DGX Spark（121 GB）。在 24 GB 显卡上，配方为：相同权重 + 更紧的 KV 预算 + ~4.5-bit KV 格式，使原生 262k 上下文仍能放得下。默认草稿器 MTP 在这里是正确的选择。

**KV 格式与 GPU 相关。** 本 fork 的 `nvfp4` / `fp8` 通道是软件打包 cache（E2M1 + E4M3 缩放，或纯 E4M3）配 Triton 在线反量化，kernel 使用 `fp8e4nv`，Triton 仅在 compute capability **≥ 8.9** 上编译；它们**不是** Blackwell tensor-core NVFP4 matmul。

| GPU | 架构 | 262k 上下文的 `CACHE_QUANT` |
|---|---|---|
| RTX 4090、GB10、Hopper+ | sm ≥ 8.9 | `nvfp4`（生成层面实测无损） |
| RTX 3090 | Ampere sm_86 | `4`（Hadamard int4，同为 ~4.5 bits/elem）。`nvfp4` 与 `fp8` 会编译失败 |

> **Windows 提示**：由于上游 wheel 不支持 `nvfp4`，Windows 上请改用 `8`（约 8.5 bits/elem）或 `8,4`。此时 262k 上下文约需 9.1 GB KV cache，在 24 GB 卡上偏紧，建议适当调低 `CONTEXT_SIZE`（start.ps1 默认 65536）。Linux / GB10 上仍按上表使用 `nvfp4` / `4`。

```bash
# .env — 24 GB 配方（MTP，默认）
GPU_MEM_GB=22            # 未设置时自动检测；显式写出更清晰
CONTEXT_SIZE=262144      # 原生上下文可完整容纳（见下方估算）
CACHE_QUANT=nvfp4        # 4090 / GB10 / Hopper+。3090 请用 CACHE_QUANT=4
# DRAFT=mtp 即默认 —— 无需额外配置、无需下载草稿
```

```bash
# .env — 备选：24 GB 下的 DFlash2
GPU_MEM_GB=22
CONTEXT_SIZE=200000
CACHE_QUANT=nvfp4        # 3090: CACHE_QUANT=4
DRAFT=dflash2
DRAFT_DIR=models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw   # 自动下载（Linux）
# CPU_CACHE_GB=16        # CPU 溢出层：尚未生效（计划中）
```

显存估算（GiB）：目标权重 14.2 + MTP 头 ~0.05 → 22 GB 预算下约剩 6.7 GB 给 KV；NVFP4 或 Hadamard-4 KV 约 18 KB/token（目标）+ 1.2 KB/token（MTP 头；只有 16 个 full-attention 层持有 KV；fp16 为 ~64 KB/token）→ 原生 262k 全部放得下，还余约 1 GB。换 DFlash2 草稿：15.6 GiB 权重 + ~24 KB/token → ~220k–262k。`CACHE_QUANT=8`（~8.5 bits）在 262k 下约需 9.1 GB，余量不大；fp16（262k 下约 17 GB）放不下。超出常驻上限时 cache 目前不会溢出；CPU spill 层计划中但尚未实现。

RTX 显卡说明（相对上述 DGX Spark 实测数据）：

- **性能受内存带宽限制（batch-1），RTX 上应该更快**：GB10 从 ~273 GB/s 级 LPDDR5x 统一内存读权重；3090/4090 从 ~1 TB/s 级 VRAM 读相同权重。接受率与质量不变，只有 tok/s 变化。尚未在 RTX 上基准测试，模型卡中的表格请视为下界
- **CPU spill（计划中，`CPU_CACHE_GB`）**：实现后（T6 跟进），桌面机 32–64 GB 系统内存可经 PCIe 备份冷 KV 页——访问慢，但把 ~220k 的硬性常驻上限变成优雅下降。当前该旋钮已接受但不生效
- **1M YaRN 配置放不下**：1M token 的 ~4.5-bit KV ≈ 19 GB，叠加上 15.6 GB 权重。只有 CPU spill 可触及，且 262k 本就是质量保真的极限——别指望 1M 余量在 24 GB 卡上有用
- **x86 也请用 fork**：上游 exllamav3 可在 CUDA 上服务该模型，但 NVFP4/FP8 KV 与 DFlash2 草稿是 fork 特性；x86 请安装 fork 的 x86 CUDA 构建。Hadamard `4` / `8` / `8,4` KV 为上游能力
- 需要超过 ~262k 的上下文？那需要 YaRN 1M 配置和超过 24 GB 卡的内存。`DRAFT=none` 也能释放草稿内存，但放弃了投机解码——`DRAFT=mtp` 在内存与速度上都已优于它

## 多卡推理（双卡 / 多卡）

`start.ps1`（Windows）支持多卡，无需改命令行——在 `.env` 里配置即可。底层透传给 exllamav3 的 `-gs/--gpu_split` 与 `-tp/--tensor_parallel` 参数。

### 1. 启动时自动检测 GPU

脚本启动时会枚举所有可见 GPU（经 `nvidia-smi`，尊重 `CUDA_VISIBLE_DEVICES`）并打印清单，例如本机 2 × RTX A4000（16 GB）：

```
检测到 2 张 GPU:
  [0] NVIDIA RTX A4000  16376 MiB (~16.0 GB, 建议预算 13 GB)
  [1] NVIDIA RTX A4000  16376 MiB (~16.0 GB, 建议预算 13 GB)
```

默认仍是**单卡模式**（只用 GPU0，预算取 `GPU_MEM_GB`，未设则按第一张卡的 `VRAM − 2 GB` 推断）——多卡必须显式开启。

### 2. 两种切分方式

| 方式 | 说明 | 配置 |
|---|---|---|
| **层切分**（默认） | 连续的 transformer 层放到不同卡上（如 0–15 层在 GPU0，16–31 层在 GPU1）。最稳、显存利用率高，无额外通信 kernel 要求 | `GPU_SPLIT=auto` 或 `13,13` |
| **张量并行** | 每层权重按列/行切到多张卡，每层前向都跨卡通信。通常延迟更低、更快，但依赖跨卡通信；与 `moe_cpu_offload/split` 互斥（本启动器未启用） | 再加 `TENSOR_PARALLEL=1` |

### 3. 三种配置写法（`.env`）

```bash
# ① 自动多卡，层切分（推荐首选）——引擎按各卡"实时空闲显存"自动分配，
#    异构卡（如 24G+16G）、有其他程序占显存时也最合理
GPU_SPLIT=auto

# ② 显式每卡预算（GB，支持小数；值的个数必须等于卡数，否则启动报错）
GPU_SPLIT=13,13          # 两张 16 GB 卡
# GPU_SPLIT=24,18.5      # 异构卡示例

# ③ 张量并行（不配 GPU_SPLIT 时自动等同 GPU_SPLIT=auto）
TENSOR_PARALLEL=1
# TP_BACKEND=native      # 默认；需要时可改 nccl
```

最终下发给引擎的参数会在启动日志中打印，可直接确认：

```
多卡模式: GPU_SPLIT=auto (exllamav3 原生 autosplit — 按各卡实时空闲显存自动分配)
启动: ... tools\serve_openai.py -m ... -gs auto -cs 65536 --host 0.0.0.0 --port 8888 -dm mtp
```

### 4. 选择参与推理的卡

用标准的 `CUDA_VISIBLE_DEVICES` 环境变量（启动器清单与 exllamav3/torch 均尊重）：

```powershell
# 只用物理卡 0 和 1（PowerShell 当前会话）
$env:CUDA_VISIBLE_DEVICES="0,1"
powershell -ExecutionPolicy Bypass -File start.ps1

# 只用物理卡 1
$env:CUDA_VISIBLE_DEVICES="1"
powershell -ExecutionPolicy Bypass -File start.ps1
```

### 5. 注意事项

- **单值 = 只用 GPU0**：`GPU_SPLIT=13`（或 `GPU_MEM_GB=13`）会被上游解析为单元素预算列表，**只使用 GPU0**。多卡必须写 `auto` 或 ≥2 个逗号分隔的值。
- **MTP / DFlash2 草稿均可与多卡组合**，无需额外配置；`DRAFT=mtp`（默认）在多卡下依然是显存性价比最高的选择。
- **KV cache 预算随卡数翻倍**：`CONTEXT_SIZE` 是所有卡上 KV cache 的总 token 容量上限，多卡下可容纳的上下文近似随卡数增长。例如 2 × 16 GB 卡 + `CACHE_QUANT=8`，可把 `CONTEXT_SIZE` 设得比单卡明显更大（按实际加载日志中 cache 占用调整）。
- **张量并行要求模型架构支持**（Qwen3.8 支持）；若引擎报 `Tensor-parallel is not currently implemented for ...`，退回层切分即可。
- **验证是否真的用上多卡**：加载完成后另开终端执行 `nvidia-smi`，应看到两张卡都有常驻显存占用；也可 `curl http://localhost:8888/health` 确认服务就绪。
- 先试 **层切分 `auto`**，追求更低延迟再试 `TENSOR_PARALLEL=1`，两种模式都能随时改 `.env` 重启切换。

## Tool calling

OpenAI 风格的 `tools` / `tool_choice` 可用（仓库内容表中提及但此前未在文档中展开），实测请求/响应形态：

```bash
curl http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-exl3-3.5bpw-wm",
    "messages": [{"role": "user", "content": "What is the weather in Lyon?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 300
  }'
```

返回标准 OpenAI `tool_calls` 数组（`finish_reason: "tool_calls"`、`message.content: null`），同时带常规的 `reasoning_content`。响应中的 `model` id 不一定与 `HF_TARGET_REPO` / `MODEL_DIR` 完全一致（实测 `MODEL_DIR=models/Qwen3.8-27B-EXL3-3.5bpw` 对应 `qwen3.8-27b-exl3-3.5bpw-wm`）——客户端应发的 id 以 `/v1/models` 返回为准，不要自行假设。

## 模型卡

- [`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw) —— 目标模型：EXL3 3.5bpw，按工作负载校准，14.2 GB
- [`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) —— 草稿模型：DFlash2 EXL3 5.0bpw，1.4 GB，接受率持平下解码吞吐 +33%

DGX Spark（GB10）实测：HumanEval 类解码 **47.5 tok/s**（T=0.6，接受 4.43 tokens/step）；tool-eval-bench hardmode @ T=1.0：**87–88 / 100**。

## 许可证

仓库代码：[MIT](LICENSE)。模型权重为 Apache-2.0 衍生（见模型卡）；exllamav3 为 MIT。
