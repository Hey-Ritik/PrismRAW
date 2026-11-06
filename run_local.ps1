$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    throw 'Create .venv and install the project extras first; see README.md.'
}
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.cache/torch'
& '.venv/Scripts/python.exe' scripts/serve.py @args
