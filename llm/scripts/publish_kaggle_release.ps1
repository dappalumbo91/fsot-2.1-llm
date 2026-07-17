# Rebuild clean package from local release and push a new Kaggle model version.
# Usage: .\llm\scripts\publish_kaggle_release.ps1 [-Notes "changelog"]
param(
  [string]$Notes = "FSOT-2.1-Instruct-0.5B release update"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $Root "llm\models\release\FSOT-2.1-Instruct-0.5B\model.safetensors"))) {
  $Root = "C:\Users\damia\Desktop\fsot 2.1 llm"
}
$Src = Join-Path $Root "llm\models\release\FSOT-2.1-Instruct-0.5B"
$Pkg = Join-Path $Root "llm\models\release\kaggle_package"
if (-not (Test-Path (Join-Path $Src "model.safetensors"))) {
  throw "Missing local release at $Src — run: python -m fsot_llm.export_release"
}
New-Item -ItemType Directory -Force -Path $Pkg | Out-Null
$files = @(
  "model.safetensors","config.json","generation_config.json",
  "tokenizer.json","tokenizer_config.json","chat_template.jinja",
  "README.md","fsot_release_meta.json","model-instance-metadata.json"
)
foreach ($f in $files) {
  $from = Join-Path $Src $f
  $alt = Join-Path $Pkg $f
  if (Test-Path $from) { Copy-Item $from (Join-Path $Pkg $f) -Force }
}
if (-not (Test-Path (Join-Path $Pkg "model-instance-metadata.json"))) {
  throw "model-instance-metadata.json missing in package — create once with kaggle models instances init"
}
Write-Host "Uploading version to damianpalumbo/fsot-21-instruct-05b/PyTorch/transformers ..."
kaggle models instances versions create `
  damianpalumbo/fsot-21-instruct-05b/PyTorch/transformers `
  -p $Pkg -n $Notes -r skip
Write-Host "Done. https://www.kaggle.com/models/damianpalumbo/fsot-21-instruct-05b/PyTorch/transformers"
