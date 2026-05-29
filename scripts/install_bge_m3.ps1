# scripts/install_bge_m3.ps1
# ============================================================
# Cài sentence-transformers + tải model BGE-M3 (1024-d, multilingual)
# về cache local. Sau khi xong, embed.py (B5) có thể chạy offline.
#
# Yêu cầu: Python 3.10+, ~5GB disk trống (model 2.3GB + temp).
# Thời gian: ~5-10 phút tuỳ tốc độ mạng.
#
# Cách chạy:
#   cd E:\legal-graph-kb
#   .\scripts\install_bge_m3.ps1
#
# Nếu PowerShell chặn vì execution policy:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_bge_m3.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # tắt progress bar (chậm trên Windows)

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " BGE-M3 setup cho legal-graph-kb" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# --- 1. Verify Python ---
Write-Host "`n[1/4] Kiem tra Python..." -ForegroundColor Yellow
$pyVersion = python --version 2>&1
Write-Host "      $pyVersion"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: khong tim thay python trong PATH" -ForegroundColor Red
    exit 1
}

# --- 2. Cai PyTorch (CPU) + sentence-transformers ---
# Neu ban co GPU NVIDIA va muon dung CUDA, xem:
#   https://pytorch.org/get-started/locally/
# va doi lenh pip o duoi cho phu hop (vd: --index-url https://download.pytorch.org/whl/cu121)
Write-Host "`n[2/4] Cai torch + sentence-transformers (CPU)..." -ForegroundColor Yellow
Write-Host "      (Thoi gian: ~2-3 phut, ~250MB tai ve)"
python -m pip install --upgrade pip 2>&1 | Out-Null
python -m pip install `
    "torch==2.4.1" `
    "sentence-transformers==3.2.1" `
    "huggingface_hub>=0.24" `
    "pyarrow==17.0.0" `
    "pandas==2.2.3" `
    "tqdm==4.66.5"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: pip install loi" -ForegroundColor Red
    exit 1
}

# --- 3. Tai model BGE-M3 ve cache HuggingFace ---
# Cache mac dinh: %USERPROFILE%\.cache\huggingface\hub
# Co the doi bang env var $env:HF_HOME truoc khi chay script nay.
Write-Host "`n[3/4] Tai model BAAI/bge-m3 (~2.3GB)..." -ForegroundColor Yellow
$cacheDir = if ($env:HF_HOME) { $env:HF_HOME } else { "$env:USERPROFILE\.cache\huggingface\hub" }
Write-Host "      Cache: $cacheDir"
Write-Host "      (Thoi gian: ~5-7 phut tuy mang)"

python -c @"
import os, sys
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        repo_id='BAAI/bge-m3',
        allow_patterns=[
            'config.json',
            'sentence_bert_config.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'sentencepiece.bpe.model',
            'modules.json',
            '1_Pooling/config.json',
            'config_sentence_transformers.json',
            'model.safetensors',
        ],
    )
    print(f'OK: model tai ve {path}')
except Exception as e:
    print(f'FAIL: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(1)
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: download model loi" -ForegroundColor Red
    exit 1
}

# --- 4. Smoke test: embed 1 cau tieng Viet ---
Write-Host "`n[4/4] Smoke test embedding..." -ForegroundColor Yellow

python -c @"
import time
from sentence_transformers import SentenceTransformer
t0 = time.monotonic()
print('Loading model (lan dau co the cham vi compile)...')
m = SentenceTransformer('BAAI/bge-m3')
print(f'  Load xong trong {time.monotonic()-t0:.1f}s')

texts = [
    'Nguoi lao dong duoc huong luong huu khi du tuoi nghi huu.',
    'Bao hiem xa hoi la su bao dam thay the thu nhap cho nguoi tham gia.',
    'Quy bao hiem xa hoi do co quan bao hiem xa hoi quan ly.',
]
t0 = time.monotonic()
embs = m.encode(texts, normalize_embeddings=True)
elapsed = time.monotonic() - t0
print(f'Encode {len(texts)} cau trong {elapsed:.2f}s ({elapsed/len(texts)*1000:.0f} ms/cau)')
print(f'Shape: {embs.shape}  (expect (3, 1024))')
print(f'Dim 1024 = {embs.shape[1] == 1024}')

# Test similarity
import numpy as np
sim = float(np.dot(embs[0], embs[1]))
print(f'Cosine(NLD-luong huu, BHXH-thu nhap) = {sim:.3f}  (expect > 0.5)')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nFAIL: smoke test loi" -ForegroundColor Red
    exit 1
}

# --- Verify dung luong cache ---
$modelDir = Get-ChildItem "$cacheDir\models--BAAI--bge-m3" -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
if ($modelDir.Sum) {
    $sizeMB = [math]::Round($modelDir.Sum / 1MB, 1)
    Write-Host "`nDung luong model tren disk: $sizeMB MB" -ForegroundColor Gray
}

Write-Host "`n==============================================" -ForegroundColor Green
Write-Host " HOAN TAT — san sang chay 'python -m offline.embed'" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Buoc tiep theo (sau khi B5 da co):" -ForegroundColor Yellow
Write-Host "  python -m offline.embed"
Write-Host ""
Write-Host "Neu muon dung GPU NVIDIA (nhanh hon 10-20x):"
Write-Host "  1. Cai CUDA toolkit"
Write-Host "  2. pip uninstall torch -y"
Write-Host "  3. pip install torch --index-url https://download.pytorch.org/whl/cu121"
Write-Host "  4. Edit .env: EMBED_DEVICE=cuda"
