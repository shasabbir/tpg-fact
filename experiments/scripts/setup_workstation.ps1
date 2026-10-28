param(
    [string]$PythonVersion = "3.13",
    [string]$TorchIndexUrl = ""
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $ExperimentRoot ".venv-workstation"

Set-Location -LiteralPath $ExperimentRoot
if (-not (Test-Path -LiteralPath $VenvRoot)) {
    py "-$PythonVersion" -m venv $VenvRoot
}
$Python = Join-Path $VenvRoot "Scripts\python.exe"
& $Python -m pip install --upgrade pip wheel setuptools
if ($TorchIndexUrl) {
    & $Python -m pip install torch torchvision --index-url $TorchIndexUrl
}
& $Python -m pip install -e ".[formal,analysis,dev]"
& $Python -m pytest
& $Python scripts/check_workstation.py `
    --storage-root $ExperimentRoot `
    --output outputs/preflight/workstation.json `
    --strict

Write-Output "Workstation environment passed: $VenvRoot"
