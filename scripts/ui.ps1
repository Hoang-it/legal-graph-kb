<#
.SYNOPSIS
    Launch the Streamlit chatbot UI for BHXH Q&A (Logic-LM + HyDE-semantic).

.DESCRIPTION
    Wrapper that sets UTF-8 encoding for the Windows console + Python I/O (so
    Vietnamese renders correctly), then runs `streamlit run ui/app.py` from the
    project root. The app calls the real pipeline — Neo4j + BGE-M3 + OpenAI +
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
try {
    python -m streamlit run ui/app.py --server.port $Port
}
finally {
    Pop-Location
    [Console]::OutputEncoding = $prevOutputEncoding
    [Console]::InputEncoding = $prevInputEncoding
}
