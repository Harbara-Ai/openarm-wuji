$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $projectRoot '.venvs/openarm-sim/Scripts/openarm-mujoco-launch.exe') --no-sheet

