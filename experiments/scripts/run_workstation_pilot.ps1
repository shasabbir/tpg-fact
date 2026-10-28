param(
    [int]$TrainClaims = 200,
    [int]$ValidationClaims = 200,
    [int]$DownloadWorkers = 12,
    [int]$TextBatchSize = 32,
    [int]$ImageBatchSize = 64
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ExperimentRoot ".venv-workstation\Scripts\python.exe"
$DataRoot = Join-Path $ExperimentRoot "data\public\mocheg"
$ProcessedRoot = Join-Path $ExperimentRoot "data\processed\mocheg_development"
. (Join-Path $PSScriptRoot "native_command_guard.ps1")

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run scripts/setup_workstation.ps1 first."
}
Set-Location -LiteralPath $ExperimentRoot

Invoke-CheckedNative -Description "strict workstation preflight" -Command {
    & $Python scripts/check_workstation.py `
        --storage-root $ExperimentRoot `
        --output outputs/preflight/workstation.json `
        --strict
}

Invoke-CheckedNative -Description "MOCHEG metadata preparation" -Command {
    & $Python scripts/prepare_mocheg.py `
        --dataset-root $DataRoot `
        --output-root $ProcessedRoot `
        --download-metadata
}

Invoke-CheckedNative -Description "MOCHEG sentence reconstruction" -Command {
    & $Python scripts/prepare_mocheg.py `
        --dataset-root $DataRoot `
        --output-root $ProcessedRoot `
        --build-sentence-text
}

Invoke-CheckedNative -Description "MOCHEG pilot media acquisition" -Command {
    & $Python scripts/prepare_mocheg.py `
        --dataset-root $DataRoot `
        --output-root $ProcessedRoot `
        --download-pilot-media `
        --max-train-media-claims $TrainClaims `
        --max-validation-media-claims $ValidationClaims `
        --max-download-workers $DownloadWorkers `
        --seed 13
}

Invoke-CheckedNative -Description "MOCHEG pilot packet construction" -Command {
    & $Python scripts/prepare_mocheg.py `
        --dataset-root $DataRoot `
        --output-root $ProcessedRoot `
        --build-pilot-packets
}

Invoke-CheckedNative -Description "pinned semantic feature extraction" -Command {
    & $Python scripts/build_mocheg_features.py `
        --backend transformers `
        --device cuda `
        --processed-root $ProcessedRoot `
        --output outputs/mocheg/pilot_features.npz `
        --text-batch-size $TextBatchSize `
        --image-batch-size $ImageBatchSize
}

Invoke-CheckedNative -Description "three-seed selector pilot" -Command {
    & $Python scripts/run_mocheg_selector_pilot.py `
        --features outputs/mocheg/pilot_features.npz `
        --output-dir outputs/mocheg/selector_pilot `
        --bootstrap-samples 10000
}

$Summary = Get-Content -LiteralPath "outputs\mocheg\selector_pilot\summary.json" -Raw | ConvertFrom-Json
Write-Output "Selector development gate: $($Summary.selector_development_gate_pass)"
Write-Output "Mean CEFES minus fixed F1: $($Summary.mean_cefes_minus_fixed_f1)"
Write-Output "The formal test remains locked. Return the selector_pilot folder for audit."
