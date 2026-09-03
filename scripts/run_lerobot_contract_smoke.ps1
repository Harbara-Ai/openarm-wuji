$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venvs/lerobot-policy/Scripts/python.exe'
if (-not (Test-Path $python)) {
    throw 'LeRobot environment missing: .venvs/lerobot-policy'
}
& $python -m unittest discover -s (Join-Path $projectRoot 'tests') -p 'test_lerobot_adapter.py' -v
