$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $projectRoot '.venvs/openarm-sim/Scripts/python.exe') `
    (Join-Path $PSScriptRoot 'build_reach_grasp_lift_model.py') `
    --project $projectRoot `
    --config (Join-Path $projectRoot 'configs/reach_grasp_lift.json') `
    --output (Join-Path $projectRoot 'outputs/reach_grasp_lift')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
