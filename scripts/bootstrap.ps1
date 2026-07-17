# Bootstrap FSOT 2.1 LLM workspace: venv + deps + smoke tests
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== FSOT 2.1 LLM bootstrap ===" -ForegroundColor Cyan
. (Join-Path $PSScriptRoot "set_env.ps1")

# Prefer CPython 3.11 for torch CUDA wheels (uv or system)
$candidates = @(
    "$env:USERPROFILE\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe",
    "$env:USERPROFILE\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe",
    "C:\Users\damia\AppData\Local\Programs\Python\Python314\python.exe"
)
$pyExe = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $pyExe = $c; break }
}
if (-not $pyExe) {
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue)?.Source
}
if (-not $pyExe) { throw "No Python interpreter found" }
Write-Host "Using interpreter: $pyExe"
& $pyExe --version

$venv = Join-Path $Root ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "Creating venv at $venv"
    & $pyExe -m venv $venv
}

$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"

& $python -m pip install --upgrade pip wheel setuptools
# Core science stack
& $pip install numpy mpmath sympy PyYAML pytest
# Torch from CUDA index, then HF from PyPI
& $pip install torch --index-url https://download.pytorch.org/whl/cu128
& $pip install transformers accelerate safetensors huggingface_hub sentencepiece protobuf

Write-Host "`n=== Hello FSOT (Python) ===" -ForegroundColor Cyan
& $python -m fsot_llm.hello_fsot

Write-Host "`n=== Build C++ hello ===" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "build_cpp.ps1")

Write-Host "`n=== Rust test + hello ===" -ForegroundColor Cyan
Push-Location (Join-Path $Root "fsot_core\rust")
try {
    cargo test
    cargo run --bin hello_fsot --quiet
} finally {
    Pop-Location
}

Write-Host "`n=== Parity report ===" -ForegroundColor Cyan
& $python (Join-Path $PSScriptRoot "parity_report.py")

Write-Host "`nBootstrap complete. Activate with: . .\scripts\activate.ps1" -ForegroundColor Green
