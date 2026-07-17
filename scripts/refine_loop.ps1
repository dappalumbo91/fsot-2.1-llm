# FSOT refine loop: archive check → curriculum → (optional train) → eval → ledger
param(
    [switch]$Train,
    [switch]$WithCoder,
    [switch]$WithVL,
    [int]$Epochs = 4
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Archive authority first
$ArchiveEnv = "I:\FSOT-Physical-Archive\set_fsot_archive_env.ps1"
if (Test-Path $ArchiveEnv) {
    # Do not cd away permanently — capture then return
    $here = Get-Location
    . $ArchiveEnv
    Set-Location $here
} else {
    Write-Warning "Archive env script missing — is I: mounted?"
}

. (Join-Path $PSScriptRoot "set_env.ps1")
$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "venv missing — run bootstrap first" }

Write-Host "=== 1. Archive + instrumentation eval ===" -ForegroundColor Cyan
& $python -m fsot_llm.eval_suite
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Instrumentation eval returned non-zero (check K5 archive)."
}

Write-Host "=== 2. Curriculum ===" -ForegroundColor Cyan
& $python -m fsot_llm.curriculum

if ($Train) {
    Write-Host "=== 3. LoRA train Coder 0.5B ===" -ForegroundColor Cyan
    & $python -m fsot_llm.train_lora --epochs $Epochs
}

Write-Host "=== 4. Model eval ===" -ForegroundColor Cyan
$evalArgs = @("-m", "fsot_llm.eval_suite")
if ($WithCoder) { $evalArgs += "--with-coder" }
if ($WithVL) { $evalArgs += "--with-vl" }
if (-not $WithCoder -and -not $WithVL) {
    Write-Host "No --WithCoder/--WithVL — skipping model eval (instrumentation already ran)."
} else {
    & $python @evalArgs
}

Write-Host "=== Refine loop done. Ledgers: llm\benchmarks\ledgers\ ===" -ForegroundColor Green
