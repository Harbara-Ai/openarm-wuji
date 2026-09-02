$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $projectRoot '.venvs/openarm-sim/Scripts/python.exe') `
    (Join-Path $PSScriptRoot 'build_combined_model.py') `
    --project $projectRoot `
    --config (Join-Path $projectRoot 'configs/openarm_v2_wuji_left.json') `
    --output (Join-Path $projectRoot 'outputs/combined')

