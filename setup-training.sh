#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv python3-pip git build-essential ca-certificates
if ! id trainer >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash trainer
fi
install -d -o trainer -g trainer /home/trainer/gemma-qlora
runuser -u trainer -- python3 -m venv /home/trainer/gemma-qlora/.venv
runuser -u trainer -- /home/trainer/gemma-qlora/.venv/bin/python -m pip install --upgrade pip
runuser -u trainer -- /home/trainer/gemma-qlora/.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
runuser -u trainer -- /home/trainer/gemma-qlora/.venv/bin/python -m pip install unsloth
runuser -u trainer -- /home/trainer/gemma-qlora/.venv/bin/python -m pip check
runuser -u trainer -- /home/trainer/gemma-qlora/.venv/bin/python /mnt/d/ModelTraining/check_training.py
runuser -u trainer -- /home/trainer/gemma-qlora/.venv/bin/python -m pip freeze > /mnt/d/ModelTraining/requirements-resolved.txt
