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

# --- 显存预算: 未设则自动 (独立显卡: VRAM/1024 - 2, 同 start.sh) -------------
$GPU_MEM_GB = Cfg 'GPU_MEM_GB' ''
if (-not $GPU_MEM_GB) {
    try { $vram = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1 }
    catch { $vram = $null }
    if ($vram -match '^\d+$') { $GPU_MEM_GB = [Math]::Max(8, [int][Math]::Floor([double]$vram / 1024) - 2) }
    else                      { $GPU_MEM_GB = 14 }
    Write-Host "GPU_MEM_GB 未设置 — 自动检测: $GPU_MEM_GB GB (可在 .env 覆盖)"
}

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
                '-gs', "$GPU_MEM_GB",
                '-cs', "$CONTEXT_SIZE",
                '--host', $BIND,
                '--port', "$PORT")
if ($CACHE_QUANT -and $CACHE_QUANT -ne 'none') { $serverArgs += @('-cq', $CACHE_QUANT) }
switch ($DRAFT) {
    'mtp'     { $serverArgs += @('-dm', 'mtp') }
    'dflash2' { $serverArgs += @('-dm', $DRAFT_DIR) }
    'none'    { $serverArgs += @('-dm', 'none') }
}
if ($CPU_CACHE_GB -gt 0) { $serverArgs += @('-ccs', "$CPU_CACHE_GB") }

Write-Host ("启动: " + ($serverArgs -join ' '))
Write-Host "就绪后可测试: curl http://localhost:$PORT/health"
& $venvPy @serverArgs
exit $LASTEXITCODE
