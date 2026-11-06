$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Create the root .venv and install backend dependencies first.' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'backend/src'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.cache/torch'
Push-Location (Join-Path $PSScriptRoot 'backend')
try { & $python scripts/serve.py @args } finally { Pop-Location }
