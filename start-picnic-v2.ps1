$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding
$OutputEncoding=[Console]::OutputEncoding
wsl -d GemmaTraining -u trainer --cd /mnt/d/ModelTraining -- /home/trainer/gemma-qlora/.venv/bin/python -u /mnt/d/ModelTraining/run_picnic_v2.py
if($LASTEXITCODE -ne 0){throw "Pipeline failed: $LASTEXITCODE"}
