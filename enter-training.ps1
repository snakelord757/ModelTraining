$ErrorActionPreference = 'Stop'
wsl -d GemmaTraining -u trainer --cd /home/trainer/gemma-qlora -- bash -c 'source .venv/bin/activate; exec bash -i'
