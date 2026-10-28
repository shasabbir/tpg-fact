param(
    [string]$PythonVersion = "3.13",
    [string]$TorchIndexUrl = ""
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $ExperimentRoot ".venv-workstation"
. (Join-Path $PSScriptRoot "native_command_guard.ps1")

Set-Location -LiteralPath $ExperimentRoot
if (-not (Test-Path -LiteralPath $VenvRoot)) {
    py "-$PythonVersion" -m venv $VenvRoot
}
$Python = Join-Path $VenvRoot "Scripts\python.exe"
Invoke-CheckedNative -Description "pip bootstrap" -Command {
    & $Python -m pip install --upgrade pip wheel setuptools
}
if ($TorchIndexUrl) {
    Invoke-CheckedNative -Description "CUDA PyTorch installation" -Command {
        & $Python -m pip install --upgrade torch torchvision --index-url $TorchIndexUrl
    }
}
Invoke-CheckedNative -Description "project dependency installation" -Command {
    & $Python -m pip install -e ".[formal,analysis,dev]"
}
Invoke-CheckedNative -Description "complete test suite" -Command {
    & $Python -m pytest
}
Invoke-CheckedNative -Description "strict workstation preflight" -Command {
    & $Python scripts/check_workstation.py `
        --storage-root $ExperimentRoot `
        --output outputs/preflight/workstation.json `
        --strict
}

Write-Output "Workstation environment passed: $VenvRoot"
