param([ValidatePattern('^[0-9]+(,[0-9]+)*$')][string]$Lengths = '2048,4096,8192,16384,32768,65536')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$OutputEncoding = [Console]::OutputEncoding
wsl -d GemmaTraining -u trainer --cd /home/trainer/gemma-qlora -- bash -c "set -o pipefail; PYTHONIOENCODING=utf-8 /home/trainer/gemma-qlora/.venv/bin/python -u /mnt/d/ModelTraining/probe_context.py --lengths $Lengths 2>&1 | tee /mnt/d/ModelTraining/context-probe-$Lengths.log"
Write-Host ('Probe exited with code: {0}' -f $LASTEXITCODE)
