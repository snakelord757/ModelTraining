param([switch]$Base,[switch]$Previous)
$ErrorActionPreference='Stop'
if($Base -and $Previous){throw 'Choose Base or Previous, not both'}
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding
$OutputEncoding=[Console]::OutputEncoding
$modelArgs=@()
if($Base){$modelArgs+='--base'}
if($Previous){$modelArgs+='--previous'}
wsl -d GemmaTraining -u trainer --cd /mnt/d/ModelTraining -- /home/trainer/gemma-qlora/.venv/bin/python -u /mnt/d/ModelTraining/picnic_chat_v2.py @modelArgs
if($LASTEXITCODE -ne 0){throw "Reference model exited: $LASTEXITCODE"}
