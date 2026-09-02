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