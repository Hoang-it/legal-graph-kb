<#
.SYNOPSIS
    Launch the Streamlit chatbot UI for BHXH Q&A (Logic-LM + HyDE-semantic).

.DESCRIPTION
    Wrapper that sets UTF-8 encoding for the Windows console + Python I/O (so
    Vietnamese renders correctly), then runs `streamlit run ui/app.py` from the
    project root. The app calls the real pipeline - Neo4j + BGE-M3 + OpenAI +
    SWI-Prolog must be available (see .env / docs/plans/ui_logic_lm_chatbot.md).

.PARAMETER Port
    Port for the Streamlit server (default 8501).

.EXAMPLE
    .\scripts\ui.ps1
    .\scripts\ui.ps1 -Port 8600
#>
param(
    [int]$Port = 8501
)

# Switch console to UTF-8 so Vietnamese renders properly
$prevOutputEncoding = [Console]::OutputEncoding
$prevInputEncoding = [Console]::InputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Force Python to use UTF-8 for stdin/stdout/stderr
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONPATH = $PSScriptRoot + '\..'

# Run from project root (so ui/app.py resolves the repo root + .streamlit/config.toml)
Push-Location (Join-Path $PSScriptRoot '..')

# Streamlit (Uvicorn) listens on IPv4 only (0.0.0.0). On Windows, 'localhost'
# resolves to ::1 (IPv6) FIRST, so the browser hangs / "can't reach this page"
# even though the server is actually up. Therefore:
#   * --server.headless true: skip the first-run 'Email' prompt AND stop
#     Streamlit from auto-opening the broken localhost URL; also avoids hanging
#     if the console cannot take input.
#   * open http://127.0.0.1:$Port (IPv4) ourselves once the server has bound.
# NOTE: keep this script ASCII-only. Windows PowerShell 5.1 reads .ps1 as the
# ANSI codepage when there is no BOM, so non-ASCII in code (e.g. an em-dash in a
# string) corrupts parsing.
$url = "http://127.0.0.1:$Port"
Write-Host ""
Write-Host "  >> Open this URL in your browser: $url" -ForegroundColor Green
Write-Host "     (Do NOT use 'localhost' - Windows sends it to IPv6 ::1; Streamlit listens on IPv4 only)" -ForegroundColor DarkGray
Write-Host ""

# Open the IPv4 URL in the default browser shortly after launch (background job
# so it fires once the server is bound, while streamlit runs in the foreground).
$opener = Start-Job -ArgumentList $url -ScriptBlock {
    param($u)
    Start-Sleep -Seconds 6
    Start-Process $u
}
try {
    python -m streamlit run ui/app.py --server.port $Port --server.headless true
}
finally {
    if ($opener) { Remove-Job $opener -Force -ErrorAction SilentlyContinue }
    Pop-Location
    [Console]::OutputEncoding = $prevOutputEncoding
    [Console]::InputEncoding = $prevInputEncoding
}
