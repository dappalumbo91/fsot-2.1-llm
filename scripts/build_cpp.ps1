# Build FSOT C++ hello with Clang (MSVC target) or cl.exe
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "set_env.ps1")

$inc = Join-Path $Root "fsot_core\cpp\include"
$src = Join-Path $Root "fsot_core\cpp\src\hello_fsot.cpp"
$outDir = Join-Path $Root "fsot_core\cpp\build"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$out = Join-Path $outDir "hello_fsot.exe"

$clang = Get-Command clang++ -ErrorAction SilentlyContinue
$cl = Get-Command cl -ErrorAction SilentlyContinue

if ($clang) {
    Write-Host "Building with clang++..."
    # Prefer MSVC-compatible flags on Windows
    & clang++ -std=c++17 -O2 -I"$inc" "$src" -o "$out"
    if ($LASTEXITCODE -ne 0) { throw "clang++ build failed" }
} elseif ($cl) {
    Write-Host "Building with cl.exe..."
    Push-Location $outDir
    try {
        & cl /nologo /EHsc /O2 /I"$inc" /Fe:"$out" "$src"
        if ($LASTEXITCODE -ne 0) { throw "cl build failed" }
    } finally {
        Pop-Location
    }
} else {
    throw "No C++ compiler on PATH (clang++ or cl). Install LLVM or open VS Build Tools dev shell."
}

Write-Host "Running $out"
& $out
