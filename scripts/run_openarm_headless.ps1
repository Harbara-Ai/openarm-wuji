$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $projectRoot '.venvs/openarm-sim/Scripts/python.exe') (Join-Path $PSScriptRoot 'openarm_smoke.py')

