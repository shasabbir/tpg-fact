param(
    [string] $Organizer,
    [string] $CacheDirectory,
    [string] $OutputDirectory,
    [int[]] $WaitForProcess = @()
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (!$Organizer) { $Organizer = Join-Path $workspace 'tmp/averimatec_shared_task_audit' }
if (!$CacheDirectory) { $CacheDirectory = Join-Path $workspace 'tmp/gemma-hub' }
if (!$OutputDirectory) { $OutputDirectory = Join-Path $workspace 'tmp/averimatec-semantic-pilot' }
$revision = '005ad3404e59d6023443cb575daa05336842228a'
$snapshot = Join-Path $CacheDirectory "models--google--gemma-3-27b-it/snapshots/$revision"
$venv = Join-Path $workspace 'experiments/.venv-workstation/Scripts'
$env:HF_HUB_CACHE = $CacheDirectory
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
# Resume the existing sequential HTTP partial files in this cache.
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '90'

foreach ($processId in $WaitForProcess) {
    $download = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($download) {
        Write-Output "Waiting for current download process $processId."
        $download.WaitForExit()
    }
}

$line = Get-Content -LiteralPath (Join-Path $workspace '.env') |
    Where-Object { $_ -match '^\s*(?:export\s+)?hf_token\s*=' } |
    Select-Object -First 1
if ($line) {
    $env:HF_TOKEN = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
}
if (!$env:HF_TOKEN) { throw 'A Hugging Face read token is required in .env (hf_token) or HF_TOKEN.' }
try {
    & (Join-Path $venv 'python.exe') -X utf8 (Join-Path $PSScriptRoot 'fetch_averimatec_gemma.py') --cache $CacheDirectory
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
}

Write-Output "Checkpoint verified. Starting the validation-only BF16 pilot."
& (Join-Path $venv 'python.exe') -X utf8 (Join-Path $PSScriptRoot 'run_averimatec_semantic_dev.py') --mode pilot --organizer $Organizer --model-dir $snapshot --output $OutputDirectory
exit $LASTEXITCODE
