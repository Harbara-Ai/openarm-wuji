$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venvs/openarm-sim/Scripts/python.exe'
$python = if (Test-Path $projectPython) { $projectPython } elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { throw 'Python not found; create .venvs/openarm-sim or add Python to PATH' }
$env:PYTHONPATH = Join-Path $projectRoot 'src'
& $python -m unittest discover -s (Join-Path $projectRoot 'tests') -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
