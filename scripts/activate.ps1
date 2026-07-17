# Activate FSOT LLM venv + env vars for the current PowerShell session
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "set_env.ps1")
$activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    . $activate
    Write-Host "venv activated: $($env:VIRTUAL_ENV)"
} else {
    Write-Warning "No .venv yet — run .\scripts\bootstrap.ps1 first"
}
