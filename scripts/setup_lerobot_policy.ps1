param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$lerobotRoot = Join-Path $projectRoot 'vendor/lerobot'
$expectedCommit = 'fa048804d05c1b965b0a5801cf205f97a9e8a3e8'

if (-not (Test-Path (Join-Path $lerobotRoot 'pyproject.toml'))) {
    throw 'Pinned LeRobot checkout missing at vendor/lerobot'
}
$actualCommit = (& git -C $lerobotRoot rev-parse HEAD).Trim()
if ($actualCommit -ne $expectedCommit) {
    throw "LeRobot commit mismatch: expected $expectedCommit, got $actualCommit"
}
if (-not $Python) {
    $Python = if (Get-Command python -ErrorAction SilentlyContinue) {
        (Get-Command python).Source
    } else {
        throw 'Python was not found on PATH; pass -Python C:\path\to\python.exe'
    }
}
if ((& $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") -ne '3.12') {
    throw 'The pinned LeRobot version requires Python 3.12'
}
$venv = Join-Path $projectRoot '.venvs/lerobot-policy'
if (-not (Test-Path (Join-Path $venv 'Scripts/python.exe'))) {
    & $Python -m venv $venv
}
$venvPython = Join-Path $venv 'Scripts/python.exe'
& $venvPython -m pip install -e $lerobotRoot
& $venvPython -m pip install -e "${projectRoot}[sim]"
& $venvPython -m unittest discover -s (Join-Path $projectRoot 'tests') -v
