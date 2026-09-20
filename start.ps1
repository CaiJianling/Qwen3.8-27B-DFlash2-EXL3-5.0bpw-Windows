# ============================================================================
#  start.ps1 — OpenAI 兼容 exllamav3 服务器启动脚本 (start.sh 的 Windows 版)
#
#  配置: 同目录 .env (可选, KEY=VALUE, 参考 .env.example)，改完重启生效。
#
#  首次运行: 自动创建 .venv -> 安装 CUDA 版 torch 2.10 (cu128) -> 安装
#  requirements.txt (exllamav3 官方预编译 wheel ~236 MB + 服务依赖, 免编译)。
#  之后每次运行直接启动服务器。
#  模型不会自动下载 — 需已就位于 MODEL_DIR (缺省 models\Qwen3.8-27B-EXL3-3.5bpw)。
#
#  支持单卡 / 多卡推理 (.env 配置):
#    GPU_SPLIT=auto               # 自动枚举所有 GPU, 按实时空闲显存分配 (推荐)
#    GPU_SPLIT=13,13              # 显式每卡预算 (GB, 支持小数) -> 多卡层切分
#    TENSOR_PARALLEL=1            # 张量并行 (不配 GPU_SPLIT 时等同 auto)
#    TP_BACKEND=native|nccl       # 张量并行后端 (默认 native)
#  另可用 CUDA_VISIBLE_DEVICES=0,1 限制参与推理的 GPU。
#
#  启动:  powershell -ExecutionPolicy Bypass -File start.ps1
#  排错:  powershell -ExecutionPolicy Bypass -File start.ps1 -SetupOnly
# ============================================================================
[CmdletBinding()]
param(
    [switch]$SetupOnly   # 只装环境、不启动服务器
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'

# --- 读取 .env (可选) -------------------------------------------------------
$cfg = @{}
$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        $t = $line.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        $i = $t.IndexOf('=')
        if ($i -gt 0) { $cfg[$t.Substring(0, $i).Trim()] = $t.Substring($i + 1).Trim().Trim('"') }
    }
}
function Cfg([string]$Name, [string]$Default) {
    if ($cfg.ContainsKey($Name) -and $cfg[$Name] -ne '') { return $cfg[$Name] } else { return $Default }
}

# --- 环境变量透传 (.env 中 start.ps1 不显式处理、但引擎/HF 可能需要的键) ----
# 这些键读入 $cfg 后如果不导出到 $env:，子进程 (python/引擎/HF) 看不到。
# HF_TOKEN 在模型/tokenizer 需要鉴权时尤其关键。
$passThroughKeys = @('HF_TOKEN','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE',
                      'HUGGING_FACE_HUB_TOKEN','TRANSFORMERS_OFFLINE',
                      'EXL3_MOE_CPU_THREADS','EXL3_MOE_CPU_SWAP')
foreach ($k in $passThroughKeys) {
    $v = Cfg $k ''
    $cur = [Environment]::GetEnvironmentVariable($k)
    if ($v -and -not $cur) { Set-Item -Path "Env:$k" -Value $v }
}

# --- 配置 (键名与 .env.example 一致) ----------------------------------------
$MODEL_DIR       = Cfg 'MODEL_DIR'       'models\Qwen3.8-27B-EXL3-3.5bpw'
$CONTEXT_SIZE    = [int](Cfg 'CONTEXT_SIZE' '65536')
# KV cache 量化位宽: 上游 -cq 只接受整数位宽 (如 '8' = fp8 KV, 或 '8,4' = k8/v4)。
# 量化 cache 走 _fns_qc dispatch 路径 (triton 后端), 需要装 triton-windows。
# 注意: 'nvfp4' 是 fork 专有格式，上游 exllamav3 不支持
$CACHE_QUANT     = Cfg 'CACHE_QUANT'     '8'
if ($CACHE_QUANT -and $CACHE_QUANT -ne 'none' -and $CACHE_QUANT -notmatch '^\d+(,\d+)?$') {
    Write-Warning "CACHE_QUANT='$CACHE_QUANT' 格式无效 (应为 '8' 或 '8,4') — 忽略此项"
    $CACHE_QUANT = ''
}
$PORT            = [int](Cfg 'PORT'      '8888')
$BIND            = Cfg 'HOST'            '0.0.0.0'
$CPU_CACHE_GB    = [double](Cfg 'CPU_CACHE_GB' '0')
# torch 版本必须与 exllamav3 Release wheel 的构建对齐 (1.4.5+cu128 要求 torch==2.10.0)，
# 且要用 cu128 索引 —— PyPI 上的 2.10.0 是 CPU 版
$TORCH_INDEX_URL = Cfg 'TORCH_INDEX_URL' 'https://download.pytorch.org/whl/cu128'
$TORCH_SPEC      = Cfg 'TORCH_SPEC'      'torch==2.10.0+cu128'

# DRAFT = mtp | dflash2 | none；兼容旧写法: 只设 DRAFT_DIR=none 或 DRAFT_DIR=<路径>
$DRAFT_DIR = Cfg 'DRAFT_DIR' ''
$DRAFT     = Cfg 'DRAFT'     ''
if (-not $DRAFT) {
    if ($DRAFT_DIR -eq 'none') { $DRAFT = 'none' }
    elseif ($DRAFT_DIR)        { $DRAFT = 'dflash2' }
    else                       { $DRAFT = 'mtp' }   # 默认: MTP 头 (无额外权重, 上下文/显存性价比最高)
}
$DRAFT = $DRAFT.ToLower()
if ($DRAFT -eq 'dflash2' -and -not $DRAFT_DIR) { $DRAFT_DIR = 'models\Qwen3.8-27B-DFlash2-EXL3-5.0bpw' }
if ($DRAFT -notin @('mtp', 'dflash2', 'none')) { throw "DRAFT 必须是 mtp / dflash2 / none (当前: $DRAFT)" }

# --- GPU 清单自动检测 (nvidia-smi; 尊重 CUDA_VISIBLE_DEVICES) ----------------
# 枚举所有 NVIDIA GPU 的物理索引/名称/显存, 供单卡默认预算推断与多卡清单展示。
# CUDA_VISIBLE_DEVICES 若为数字索引形式 (如 "0,1") 则过滤清单; UUID 形式不解析
# (exllamav3/torch 原生支持, auto 模式无需本脚本处理, 清单仅作展示)。
$gpuList = @()
try {
    $smiRows = @(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>$null)
} catch { $smiRows = @() }
foreach ($r in $smiRows) {
    $parts = $r.Split(',', 3)
    if ($parts.Count -ge 3 -and $parts[0].Trim() -match '^\d+$' -and $parts[2].Trim() -match '^\d+$') {
        $vramMb = [int]$parts[2].Trim()
        $gpuList += [pscustomobject]@{
            idx    = [int]$parts[0].Trim()
            name   = $parts[1].Trim()
            vramMb = $vramMb
            budget = [Math]::Max(8, [int][Math]::Floor($vramMb / 1024.0) - 2)
        }
    }
}
$cvd = $env:CUDA_VISIBLE_DEVICES
if ($cvd) {
    $visIdx = @(); $allNumeric = $true
    foreach ($t in ($cvd -split ',')) {
        $tt = $t.Trim()
        if ($tt -match '^\d+$') { $visIdx += [int]$tt } else { $allNumeric = $false }
    }
    if ($allNumeric -and $visIdx.Count -gt 0) { $gpuList = @($gpuList | Where-Object { $visIdx -contains $_.idx }) }
}
if ($gpuList.Count -gt 0) {
    Write-Host "检测到 $($gpuList.Count) 张 GPU:"
    foreach ($g in $gpuList) {
        Write-Host ("  [{0}] {1}  {2} MiB (~{3} GB, 建议预算 {4} GB)" -f `
            $g.idx, $g.name, $g.vramMb, [Math]::Round($g.vramMb / 1024.0, 1), $g.budget)
    }
} else {
    Write-Warning 'nvidia-smi 未检测到 GPU — 自动检测不可用 (仍可在 .env 显式设置 GPU_MEM_GB / GPU_SPLIT; 无卡时引擎会自行报错)'
}

# --- 单卡显存预算: 未设则用第一张可见 GPU (公式同 start.sh: VRAM/1024 - 2) ----
$GPU_MEM_GB = Cfg 'GPU_MEM_GB' ''
if (-not $GPU_MEM_GB) {
    if ($gpuList.Count -gt 0) {
        $g0 = $gpuList[0]
        $GPU_MEM_GB = $g0.budget
        Write-Host "GPU_MEM_GB 未设置 — 按 GPU [$($g0.idx)] $($g0.name) 自动检测: $GPU_MEM_GB GB (可在 .env 覆盖)"
    } else {
        $GPU_MEM_GB = 14
        Write-Warning 'GPU_MEM_GB 未设置且未检测到 GPU — 使用回退值 14 GB'
    }
}

# --- 多卡 (双卡 / 多卡) 推理 ------------------------------------------------
# GPU_SPLIT      : 每张卡显存预算 (GB, 支持小数), 三种写法:
#                    未设置  -> 单卡 (GPU0, 预算取 GPU_MEM_GB)
#                    auto    -> exllamav3 原生 autosplit: 自动枚举所有可见 GPU,
#                               按各卡"实时空闲显存"分配 (层切分/TP 均可; 异构卡、
#                               有其他进程占用显存时也最合理)
#                    "13,13" -> 显式多卡预算 (值数 = 卡数); 注意单值 "13" 按上游
#                               语义只使用 GPU0, 多值才会启用多卡
# TENSOR_PARALLEL: 1/true/on/yes 启用张量并行 (默认层切分); 未设 GPU_SPLIT 时
#                  自动等同 GPU_SPLIT=auto。张量并行与 MoE CPU offload 互斥
#                  (本启动器未启用 offload, 无影响)。
# TP_BACKEND     : TP 后端, 'native' (默认) 或 'nccl'。
$GPU_SPLIT       = Cfg 'GPU_SPLIT'       ''
$TENSOR_PARALLEL = Cfg 'TENSOR_PARALLEL' ''
$TP_BACKEND      = Cfg 'TP_BACKEND'      ''

# 解析 TENSOR_PARALLEL 布尔值
$tpEnabled = $false
if ($TENSOR_PARALLEL) {
    $v = $TENSOR_PARALLEL.ToLower()
    if     ($v -in @('1','true','on','yes'))  { $tpEnabled = $true }
    elseif ($v -in @('0','false','off','no')) { $tpEnabled = $false }
    else { throw "TENSOR_PARALLEL='$TENSOR_PARALLEL' 无效 (应为 1/0/true/false/on/off/yes/no)" }
}
if ($TP_BACKEND -and $TP_BACKEND -notin @('native','nccl')) {
    throw "TP_BACKEND='$TP_BACKEND' 无效 (应为 'native' 或 'nccl')"
}

# 解析 GPU_SPLIT -> 模式: single | auto | explicit ($splitTokens 保留原始字符串)
$gsMode = 'single'
$splitTokens = @()
if ($GPU_SPLIT) {
    if ($GPU_SPLIT.Trim().ToLower() -eq 'auto') {
        $gsMode = 'auto'
    } else {
        foreach ($p in ($GPU_SPLIT -split ',')) {
            $t = $p.Trim()
            if ($t -notmatch '^\d+(\.\d+)?$') {
                throw "GPU_SPLIT='$GPU_SPLIT' 格式无效 (应为 'auto' 或逗号分隔的每卡 GB 预算, 如 '13,13' / '24,18.5')"
            }
            $splitTokens += $t
        }
        if ($splitTokens.Count -ge 2) { $gsMode = 'explicit' }
        # 单值仍是 single: 上游 model_init 把单值解析为单元素列表, 只会使用 GPU0
    }
}
# TP 至少需要 2 张卡: 未显式给预算 -> auto; 只给 1 个值 -> 报错
if ($tpEnabled -and $gsMode -eq 'single') {
    if ($GPU_SPLIT) {
        throw "TENSOR_PARALLEL=1 需要 >=2 张 GPU, 但 GPU_SPLIT='$GPU_SPLIT' 只指定了 1 个预算 (多卡请写 '13,13' 或 'auto')"
    }
    $gsMode = 'auto'
}
# 显式预算卡数 > 实际可见卡数 -> 直接报错 (auto 模式交给引擎, 不做此检查)
if ($gsMode -eq 'explicit' -and $gpuList.Count -gt 0 -and $splitTokens.Count -gt $gpuList.Count) {
    throw "GPU_SPLIT 指定了 $($splitTokens.Count) 张卡, 但只检测到 $($gpuList.Count) 张可见 GPU (索引: $($gpuList.idx -join ',')); 可用 CUDA_VISIBLE_DEVICES 调整可见卡"
}

# 决定传给 serve_openai.py -gs 的值
switch ($gsMode) {
    'auto' {
        $gsValue = 'auto'
        Write-Host '多卡模式: GPU_SPLIT=auto (exllamav3 原生 autosplit — 按各卡实时空闲显存自动分配)'
    }
    'explicit' {
        $gsValue = $splitTokens -join ','
        Write-Host "多卡模式: 显式每卡预算 $gsValue GB ($($splitTokens.Count) 张 GPU)"
    }
    'single' {
        if ($splitTokens.Count -eq 1) {
            $gsValue = $splitTokens[0]
            Write-Host "单卡模式: GPU0 预算 $gsValue GB (GPU_SPLIT 单值; 要用多卡请写 '$gsValue,$gsValue' 或 'auto')"
        } else {
            $gsValue = "$GPU_MEM_GB"
            Write-Host "单卡模式: GPU0 预算 $gsValue GB (多卡设 GPU_SPLIT=auto 或 GPU_SPLIT=预算,预算)"
        }
    }
}
if ($tpEnabled) { Write-Host "张量并行: 启用 (后端: $(if ($TP_BACKEND) { $TP_BACKEND } else { 'native' }))" }
elseif ($gsMode -ne 'single') { Write-Host '张量并行: 关闭 (默认层切分 layer-split)' }

# --- 首次引导: 创建 venv + 安装依赖 -----------------------------------------
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    Write-Host '== 首次运行: 创建虚拟环境 .venv (仅一次) =='
    python -m venv .venv
    if (-not (Test-Path $venvPy)) { throw '.venv 创建失败 — 请确认 python (3.10+) 可用' }
}
# venv 保持在 PATH 最前 (与引擎运行相关的工具优先命中)
$env:PATH = "$PSScriptRoot\.venv\Scripts;$env:PATH"

$depsOk = (& $venvPy -c "import importlib.util as u; import sys; sys.exit(0 if all(u.find_spec(m) for m in ('torch','exllamav3','aiohttp','huggingface_hub')) else 1)")
if ($depsOk -ne 0) {
    Write-Host '== 安装依赖 (首次较慢: torch ~2.5 GB + exllamav3 wheel ~236 MB) =='
    & $venvPy -c "import importlib.util as u; import sys; sys.exit(0 if u.find_spec('torch') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   [1/2] $TORCH_SPEC (CUDA, 索引: $TORCH_INDEX_URL) ..."
        # 清空全局 extra-index-url，避免镜像源混入 CPU 版 torch
        $env:PIP_EXTRA_INDEX_URL = ''
        & $venvPy -m pip install $TORCH_SPEC --index-url $TORCH_INDEX_URL
        Remove-Item Env:\PIP_EXTRA_INDEX_URL -ErrorAction SilentlyContinue
        if ($LASTEXITCODE -ne 0) { throw "torch 安装失败 (索引 $TORCH_INDEX_URL) — 可在 .env 换 TORCH_INDEX_URL / TORCH_SPEC" }
    }
    Write-Host '   [2/2] requirements.txt (exllamav3 预编译 wheel + 服务依赖) ...'
    & $venvPy -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败 (详见上方 pip 输出)' }
    Write-Host '== 依赖安装完成 =='
}

# --- 补丁: exllamav3 1.4.5 在 Windows 无 triton 时的两个 bug ----------------
# Bug 1: dsa_triton.py 的 kernel 定义在 if has_triton: 块内, 无 triton 时
#        bc_dsa.py 的 top-level import 会崩溃 (非 DeepSeek-V4 模型不该触发)
# Bug 2: torch.py 的 fn_torch_sdpa_fallback_cache 要求 dim >= 512 (bighead),
#        普通 head dim (如 128) 在无 triton 时没有可用 attention 后端
# 只在尚未打补丁时执行 (幂等; 重装 exllamav3 后自动重打)
$dsaTriton = Join-Path $PSScriptRoot '.venv\Lib\site-packages\exllamav3\modules\attention_fn\dsa_triton.py'
$torchFn   = Join-Path $PSScriptRoot '.venv\Lib\site-packages\exllamav3\modules\attention_fn\torch.py'
if ((Test-Path $dsaTriton) -and -not (Select-String -Path $dsaTriton -Pattern 'No triton: define placeholder' -Quiet)) {
    $c = Get-Content $dsaTriton -Raw
    $c = $c -replace '(        tl\.store\(out \+ r \* K_pad \+ offs, v, mask = offs < K_pad\)\n\ndef dsa_attn\()', "`$1`nelse:`n    # No triton: define placeholder names so bc_dsa.py's top-level import`n    # does not crash at module load time (DSA paths are never used by`n    # non-DeepSeek-V4 models like Qwen3.8)`n    _dsa_attn_kernel = None`n    _dsa_attn_split_kernel = None`n    _dsa_attn_combine_kernel = None`n    _dsa_indexer_kernel = None`n    _dsa_indexer_fewq_kernel = None`n    _dsa_pool_update_kernel = None`n    _dsa_pool_expand_kernel = None`n`ndef dsa_attn("
    Set-Content -Path $dsaTriton -Value $c -NoNewline
    Write-Host '已打补丁: dsa_triton.py (无 triton 时的 import 修复)'
}
if ((Test-Path $torchFn) -and (Select-String -Path $torchFn -Pattern 'args\.dim < 512' -Quiet)) {
    $c = Get-Content $torchFn -Raw
    $c = $c -replace '        args\.dim < 512 or\n', ''
    Set-Content -Path $torchFn -Value $c -NoNewline
    Write-Host '已打补丁: torch.py (SDPA fallback 支持所有 head dim)'
}
if ($SetupOnly) { Write-Host '-SetupOnly: 环境就绪，跳过启动。'; exit 0 }

# --- Triton 缓存目录 (避免 %USERPROFILE%\.triton 权限问题) -----------------
$env:TRITON_CACHE_DIR = Join-Path $PSScriptRoot '.triton-cache'
New-Item -ItemType Directory -Path $env:TRITON_CACHE_DIR -Force | Out-Null

# --- 模型检查 (不自动下载) ---------------------------------------------------
if (-not [IO.Path]::IsPathRooted($MODEL_DIR)) { $MODEL_DIR = Join-Path $PSScriptRoot $MODEL_DIR }
if (-not (Test-Path (Join-Path $MODEL_DIR 'config.json'))) { throw "模型目录缺少 config.json: $MODEL_DIR (请先下载模型)" }
if (-not (Get-ChildItem $MODEL_DIR -Filter *.safetensors -ErrorAction SilentlyContinue)) { throw "模型目录缺少 *.safetensors: $MODEL_DIR" }
Write-Host "目标模型: $MODEL_DIR (已就位，跳过下载)"

# 上下文超过原生 262144 需切换 YaRN 配置 (同 start.sh)
if ($CONTEXT_SIZE -gt 262144) {
    $yarnCfg = Join-Path $MODEL_DIR 'config.yarn-1m.json'
    $mainCfg = Join-Path $MODEL_DIR 'config.json'
    if ((Test-Path $yarnCfg) -and -not (Select-String -Path $mainCfg -Pattern 'rope_scaling' -Quiet)) {
        Copy-Item $yarnCfg $mainCfg -Force
        Write-Host "CONTEXT_SIZE > 262k: config.json 已切换为 YaRN 1M 变体"
    }
}
if ($DRAFT -eq 'dflash2') {
    if (-not [IO.Path]::IsPathRooted($DRAFT_DIR)) { $DRAFT_DIR = Join-Path $PSScriptRoot $DRAFT_DIR }
    if (-not (Test-Path (Join-Path $DRAFT_DIR 'config.json'))) { throw "DFlash2 草稿模型缺失: $DRAFT_DIR (请先下载)" }
}

# --- 组装 serve_openai.py 启动参数 ------------------------------------------
$serverArgs = @('-u', (Join-Path $PSScriptRoot 'tools\serve_openai.py'),
                '-m', $MODEL_DIR,
                '-gs', $gsValue,
                '-cs', "$CONTEXT_SIZE",
                '--host', $BIND,
                '--port', "$PORT")
if ($tpEnabled)                      { $serverArgs += @('-tp') }
if ($TP_BACKEND)                    { $serverArgs += @('-tpb', $TP_BACKEND) }
if ($CACHE_QUANT -and $CACHE_QUANT -ne 'none') { $serverArgs += @('-cq', $CACHE_QUANT) }
switch ($DRAFT) {
    'mtp'     { $serverArgs += @('-dm', 'mtp') }
    'dflash2' { $serverArgs += @('-dm', $DRAFT_DIR) }
    'none'    { $serverArgs += @('-dm', 'none') }
}
if ($CPU_CACHE_GB -gt 0) { $serverArgs += @('-ccs', "$CPU_CACHE_GB") }

# --- 配置摘要 (让用户一眼确认 .env 各字段是否生效) -------------------------
Write-Host '== 配置摘要 (来源: .env 或默认值) =='
Write-Host "  MODEL_DIR     = $MODEL_DIR"
Write-Host "  CONTEXT_SIZE  = $CONTEXT_SIZE"
Write-Host "  CACHE_QUANT   = $(if ($CACHE_QUANT) { $CACHE_QUANT } else { 'none' })"
Write-Host "  PORT          = $PORT"
Write-Host "  HOST          = $BIND"
Write-Host "  DRAFT         = $DRAFT$(if ($DRAFT -eq 'dflash2') { " ($DRAFT_DIR)" })"
Write-Host "  GPU_SPLIT     = $gsValue$(if ($tpEnabled) { ' + TP' } elseif ($gsMode -ne 'single') { ' (层切分)' })"
Write-Host "  CPU_CACHE_GB  = $CPU_CACHE_GB"
Write-Host "  HF_TOKEN      = $(if ($env:HF_TOKEN) { '已设置 (' + $env:HF_TOKEN.Substring(0, [Math]::Min(8, $env:HF_TOKEN.Length)) + '...)' } else { '未设置' })"
Write-Host "=========================================="

Write-Host ("启动: " + ($serverArgs -join ' '))
Write-Host "就绪后可测试: curl http://localhost:$PORT/health"
& $venvPy @serverArgs
exit $LASTEXITCODE
