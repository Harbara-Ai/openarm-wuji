$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
& (Join-Path $PSScriptRoot 'build_reach_grasp_lift_model.ps1')
if (-not $?) { throw 'Reach-Grasp-Lift model build failed' }
& (Join-Path $projectRoot '.venvs/openarm-sim/Scripts/python.exe') `
    (Join-Path $PSScriptRoot 'grasp_smoke.py') `
    --model (Join-Path $projectRoot 'outputs/reach_grasp_lift/reach_grasp_lift.mjb') `
    --config (Join-Path $projectRoot 'configs/reach_grasp_lift.json') `
    --synergies (Join-Path $projectRoot 'configs/wuji_hand_left_synergies.json') `
    --output (Join-Path $projectRoot 'outputs/reach_grasp_lift')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
