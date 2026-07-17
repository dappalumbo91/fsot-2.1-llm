# FSOT 2.1 LLM — environment (archive master + this workspace)
# Dot-source:  . .\scripts\set_env.ps1

$ErrorActionPreference = "Stop"
# scripts/ lives one level under the workspace root
$Workspace = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Workspace "README.md"))) {
    throw "FSOT LLM workspace not found above $PSScriptRoot"
}

$ArchiveRoot = if ($env:FSOT_ARCHIVE_ROOT) { $env:FSOT_ARCHIVE_ROOT } else { "I:\FSOT-Physical-Archive" }
$LeanHub = Join-Path $ArchiveRoot "02_FSOT-2.1-Lean-Full"
$PublicData = Join-Path $ArchiveRoot "03_FSOT-PublicData"
$Compute = Join-Path $LeanHub "vendor\fsot_compute.py"

$env:FSOT_LLM_ROOT = $Workspace
$env:FSOT_ARCHIVE_ROOT = $ArchiveRoot
$env:FSOT_CANONICAL_ARCHIVE = "1"
$env:FSOT_PORTABLE = "1"
$env:FSOT_EXTERNAL_DATA_ROOT = $PublicData
$env:FSOT_ANOMALY_CACHE_ROOT = Join-Path $PublicData "anomaly_observables"
$env:FSOT_COMPUTE_PATH = $Compute

# External training / SOTA benchmark packs (D: drive)
$TrainData = if (Test-Path "D:\training data") { "D:\training data" } else { $env:FSOT_TRAINING_DATA_ROOT }
$BenchRoot = if (Test-Path "D:\FSOT_Benchmarks") { "D:\FSOT_Benchmarks" } else { $env:FSOT_BENCHMARKS_ROOT }
if ($TrainData) { $env:FSOT_TRAINING_DATA_ROOT = $TrainData }
if ($BenchRoot) { $env:FSOT_BENCHMARKS_ROOT = $BenchRoot }

# Python path for package + bridge
$env:PYTHONPATH = @(
    (Join-Path $Workspace "llm\python"),
    (Join-Path $Workspace "fsot_core\python"),
    $env:PYTHONPATH
) -join ";"


# Toolchain helpers (MSVC + Clang) if present
$Clang = "C:\Program Files\LLVM\bin"
if (Test-Path $Clang) {
    if ($env:PATH -notlike "*$Clang*") { $env:PATH = "$Clang;$env:PATH" }
}
$MsvcHost = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
if (Test-Path $MsvcHost) {
    $latest = Get-ChildItem $MsvcHost -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($latest) {
        $bin = Join-Path $latest.FullName "bin\Hostx64\x64"
        if ((Test-Path $bin) -and ($env:PATH -notlike "*$bin*")) {
            $env:PATH = "$bin;$env:PATH"
        }
    }
}

Write-Host "FSOT 2.1 LLM env:"
Write-Host "  FSOT_LLM_ROOT     = $env:FSOT_LLM_ROOT"
Write-Host "  FSOT_ARCHIVE_ROOT = $env:FSOT_ARCHIVE_ROOT"
Write-Host "  FSOT_COMPUTE_PATH = $env:FSOT_COMPUTE_PATH"
Write-Host "  compute exists    = $(Test-Path $Compute)"
Write-Host "  TRAINING_DATA     = $env:FSOT_TRAINING_DATA_ROOT (exists=$(if ($env:FSOT_TRAINING_DATA_ROOT) { Test-Path $env:FSOT_TRAINING_DATA_ROOT } else { $false }))"
Write-Host "  BENCHMARKS        = $env:FSOT_BENCHMARKS_ROOT (exists=$(if ($env:FSOT_BENCHMARKS_ROOT) { Test-Path $env:FSOT_BENCHMARKS_ROOT } else { $false }))"
