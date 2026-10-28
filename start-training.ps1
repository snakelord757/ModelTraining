param([int]$MaxSteps = -1, [double]$Epochs = 1, [int]$MaxLength = 12288,
      [int]$TrainSamples = 0,
      [string]$Resume = '', [switch]$EvaluateOnly)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$OutputEncoding = [Console]::OutputEncoding
# Foreground terminal displays the Trainer progress bar, elapsed time and ETA.
$pipelineArgs = @('--max-steps', $MaxSteps, '--epochs', $Epochs, '--max-length', $MaxLength, '--train-samples', $TrainSamples)
if ($Resume) { $pipelineArgs += @('--resume', $Resume) }
if ($EvaluateOnly) { $pipelineArgs += '--evaluate-only' }
wsl -d GemmaTraining -u trainer --cd /home/trainer/gemma-qlora -- /home/trainer/gemma-qlora/.venv/bin/python -u /mnt/d/ModelTraining/run_training_pipeline.py @pipelineArgs
if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
