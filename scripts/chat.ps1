<#
.SYNOPSIS
    Launch interactive chat with the Legal KG RAG system.

.DESCRIPTION
    Wrapper script that sets UTF-8 encoding for Windows console + Python I/O,
    then runs `python -m runtime.chat`. Without these env vars, Vietnamese input
    via PowerShell may be garbled.

.PARAMETER TopK
    Number of Clauses to retrieve per question (default 8).

.PARAMETER NoVerify
    Skip citation verification against DB (faster).

.PARAMETER NoRich
    Disable rich formatting (use plain text).

.EXAMPLE
    .\scripts\chat.ps1
    .\scripts\chat.ps1 -TopK 12
    .\scripts\chat.ps1 -NoVerify -NoRich
#>
param(
    [int]$TopK = 8,
    [switch]$NoVerify,
    [switch]$NoRich
)

# Switch console to UTF-8 (chcp 65001) to render Vietnamese properly
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

# Build args
$pyArgs = @('-m', 'runtime.chat', '--top-k', $TopK)
if ($NoVerify) { $pyArgs += '--no-verify' }
if ($NoRich)   { $pyArgs += '--no-rich' }

# Run from project root
Push-Location (Join-Path $PSScriptRoot '..')
try {
    python @pyArgs
}
finally {
    Pop-Location
    [Console]::OutputEncoding = $prevOutputEncoding
    [Console]::InputEncoding = $prevInputEncoding
}
