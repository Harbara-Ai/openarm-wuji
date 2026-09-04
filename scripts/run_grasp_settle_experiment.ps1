$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venvs/openarm-sim/Scripts/python.exe'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$model = Join-Path $projectRoot 'outputs/reach_grasp_lift/reach_grasp_lift.mjb'
$config = Join-Path $projectRoot 'configs/reach_grasp_lift.json'
$synergies = Join-Path $projectRoot 'configs/wuji_hand_left_synergies.json'
$output = Join-Path $projectRoot 'outputs/reach_grasp_lift'

& (Join-Path $PSScriptRoot 'build_reach_grasp_lift_model.ps1')
if (-not $?) { throw 'Reach-Grasp-Lift model build failed' }

& $python (Join-Path $PSScriptRoot 'lift_smoke.py') `
    --model $model `
    --config $config `
    --synergies $synergies `
    --output $output `
    --seed 7 `
    --expected-outcome drop
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python (Join-Path $PSScriptRoot 'record_lift_episode.py') `
    --model $model `
    --config $config `
    --synergies $synergies `
    --output $output `
    --seed 7 `
    --expected-outcome drop
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
