$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
& (Join-Path $projectRoot '.venvs/openarm-sim/Scripts/python.exe') `
    (Join-Path $PSScriptRoot 'combined_control_smoke.py') `
    --model (Join-Path $projectRoot 'outputs/combined/openarm_v2_wuji_left.mjb') `
    --synergies (Join-Path $projectRoot 'configs/wuji_hand_left_synergies.json') `
    --output (Join-Path $projectRoot 'outputs/combined')
