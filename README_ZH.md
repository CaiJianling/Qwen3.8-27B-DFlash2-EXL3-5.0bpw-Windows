# 环境：Python3.13，CUDA13.3，torch2.10.0+cu128
# 软件下载
# github加速：
```
git clone https://gh-proxy.org/https://github.com/CaiJianling/Qwen3.8-27B-DFlash2-EXL3-5.0bpw-Windows.git
```
# 模型下载
# hf加速：
```
$env:HF_ENDPOINT="https://hf-cdn.sufy.com"
hf download Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw `
 --local-dir ./Qwen3.8-27B-EXL3-3.5bpw `
 --max-workers 6
```
# 安装Visual Studio Installer
# 安装Community 20xx版本
# 安装使用C++ 的桌面开发，需要用到C++ 编译
# 打开x64 Native Tools Command Prompt for VS 20xx，执行以下命令：
```
where cl
```
# 将安装地址放到PATH中，确认终端可以调用到`cl`
# 执行 `start.ps1` 脚本
```
./start.ps1
```
# 多卡推理（双卡/多卡，详见 README.md「多卡推理」章节）
# 启动时会自动列出检测到的 GPU（索引/名称/显存）。在 `.env` 中配置：
```
GPU_SPLIT=auto            # 推荐：引擎按各卡实时空闲显存自动分配（层切分）
# GPU_SPLIT=13,13         # 或显式指定每卡预算(GB, 支持小数)，值数=卡数
# TENSOR_PARALLEL=1       # 张量并行（默认层切分；不配 GPU_SPLIT 时自动等同 auto）
```
# 注意：GPU_SPLIT=13（单值）只使用 GPU0；多卡必须写 auto 或 13,13 这样的多值。
# 只让部分物理卡参与推理：
```
$env:CUDA_VISIBLE_DEVICES="0,1"
./start.ps1
```