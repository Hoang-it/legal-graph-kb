<#
.SYNOPSIS
    Install dependencies for Step 5 (embedding) - local BGE-M3 1024-d model.

.DESCRIPTION
    Installs torch + sentence-transformers + pyarrow + tqdm, then pre-downloads
    BAAI/bge-m3 model (~2.3 GB) so embed.py can run offline.

.PARAMETER Backend
    'cpu' (default), 'cu121', 'cu118', or 'cu124' - choose torch wheel.
    For RTX 3050: 'cu121' is recommended (~5-10x faster than CPU).

.PARAMETER SkipModelDownload
    Skip pre-downloading the model (only install Python deps).

.PARAMETER VerifyOnly
    Don't install anything, just run verify_b5.py.

.EXAMPLE
    .\scripts\install_b5.ps1                 # CPU
    .\scripts\install_b5.ps1 -Backend cu121  # GPU CUDA 12.1 (recommended for RTX 3050)
    .\scripts\install_b5.ps1 -SkipModelDownload
    .\scripts\install_b5.ps1 -VerifyOnly

.NOTES
    Disk usage:
    - torch CPU only        : ~700 MB
    - torch CUDA            : ~2.5 GB
    - sentence-transformers : ~150 MB
    - BGE-M3 model          : ~2.3 GB
    Total estimate (CUDA)   : ~5 GB

    Note: this script uses English-only messages so it works correctly with
    Windows PowerShell 5.1 which reads .ps1 files as ANSI (Windows-1252).
#>
param(
    [ValidateSet('cpu', 'cu118', 'cu121', 'cu124')]
    [string]$Backend = 'cpu',
    [switch]$SkipModelDownload,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

# Ensure Python child processes write UTF-8 (so verify script prints Vietnamese OK)
$env:PYTHONIOENCODING = 'utf-8'
# Suppress noisy warning about HF cache symlinks on Windows
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

# ----- VerifyOnly mode -----
if ($VerifyOnly) {
    Write-Step "Verify only"
    python "$PSScriptRoot\verify_b5.py"
    exit $LASTEXITCODE
}

# ----- Sanity: should be in project root -----
if (-not (Test-Path "src/schema.py")) {
    Write-Warning "This script should run from project root (E:\legal-graph-kb)."
    Write-Warning "Current location: $(Get-Location)"
}

# ----- 1. Check Python -----
Write-Step "Step 1/4 - Check Python"
$pyVersion = (python --version) 2>&1
Write-Host "  $pyVersion"

# ----- 2. Install torch (>= 2.6 required by sentence-transformers for safetensors-only) -----
Write-Step "Step 2/4 - Install torch (backend = $Backend)"
# torch 2.6+ required to bypass CVE-2025-32434 (older torch.load is blocked).
# cu121 wheel index stops at torch 2.5.1, so cu121 callers get cu124 wheels instead.
$torchSpec = "torch==2.6.0"
switch ($Backend) {
    'cpu'   { $indexUrl = 'https://download.pytorch.org/whl/cpu' }
    'cu118' { $indexUrl = 'https://download.pytorch.org/whl/cu118' }
    'cu121' {
        Write-Host "  Note: cu121 wheels max out at torch 2.5.1; using cu124 wheels for torch 2.6.0."
        $indexUrl = 'https://download.pytorch.org/whl/cu124'
    }
    'cu124' { $indexUrl = 'https://download.pytorch.org/whl/cu124' }
}
Write-Host "  pip install $torchSpec --index-url $indexUrl --upgrade"
pip install $torchSpec --index-url $indexUrl --upgrade
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

# ----- 3. Install sentence-transformers + utils -----
# Pin datasets==3.0.1: newer datasets need pa.json_() which requires pyarrow>=19,
# but pyarrow 22+ has Windows DLL access violations. This combo is known-good.
Write-Step "Step 3/4 - Install sentence-transformers, datasets, pyarrow, pandas, tqdm"
pip install `
    "sentence-transformers==3.2.1" `
    "datasets==3.0.1" `
    "pyarrow==17.0.0" `
    "pandas==2.2.3" `
    "tqdm==4.66.5"
if ($LASTEXITCODE -ne 0) { throw "sentence-transformers install failed" }

# ----- 4. Pre-download BGE-M3 model -----
if (-not $SkipModelDownload) {
    Write-Step "Step 4/4 - Pre-download BAAI/bge-m3 (~2.3 GB)"
    Write-Host "  (First-time download takes 3-10 min depending on network speed)"
    Write-Host "  Cached at: ~\.cache\huggingface"
    python -c @"
import time
from sentence_transformers import SentenceTransformer
t = time.time()
print('Downloading BAAI/bge-m3 ...')
m = SentenceTransformer('BAAI/bge-m3')
print(f'OK - loaded in {time.time()-t:.1f}s, dim = {m.get_sentence_embedding_dimension()}')
"@
    if ($LASTEXITCODE -ne 0) { throw "Model pre-download failed" }
} else {
    Write-Step "Step 4/4 - Skipped (--SkipModelDownload)"
}

# ----- 5. Verify -----
Write-Step "Verify"
python "$PSScriptRoot\verify_b5.py"
$verifyCode = $LASTEXITCODE

# ----- Final summary -----
Write-Step "Done"
if ($verifyCode -eq 0) {
    Write-Host "OK - everything ready for Step 5." -ForegroundColor Green
    Write-Host "Run: python -m src.embed"
} else {
    Write-Warning "Verify returned exit code $verifyCode. See log above."
}
exit $verifyCode
