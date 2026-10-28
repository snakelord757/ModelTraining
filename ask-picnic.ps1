param([switch]$Base)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding
$OutputEncoding=[Console]::OutputEncoding
$modelArgs=@()
if($Base){$modelArgs+='--base'}
wsl -d GemmaTraining -u trainer --cd /mnt/d/ModelTraining -- /home/trainer/gemma-qlora/.venv/bin/python -u /mnt/d/ModelTraining/picnic_inference.py @modelArgs
if($LASTEXITCODE -ne 0){throw "Reference model exited: $LASTEXITCODE"}
