param(
    [string]$Distro = 'Ubuntu-22.04',
    [Parameter(Mandatory = $true)][string]$WslPython,
    [Parameter(Mandatory = $true)][string]$WujiSource
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$linuxProject = (wsl.exe -d $Distro -- wslpath -a ($projectRoot -replace '\\', '/')).Trim()
wsl.exe -d $Distro -- $WslPython `
    "$linuxProject/scripts/wuji_retarget_smoke.py" `
    --source $WujiSource `
    --output "$linuxProject/outputs/wuji" `
    --hand left `
    --max-frames 300
