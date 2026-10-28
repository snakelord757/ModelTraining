"""Run base evaluation, training and adapter evaluation in separate GPU processes."""
import argparse
import datetime
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from evaluate_literature import ROOT, read_json, write_json, render_report
from model_settings import OUTPUT, MAX_LENGTH, TRAIN_SAMPLES, MODEL_ID, require_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--max-length', type=int, default=MAX_LENGTH)
    p.add_argument('--train-samples', type=int, default=TRAIN_SAMPLES)
    p.add_argument('--resume')
    p.add_argument('--output', default=OUTPUT)
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args()
    manifest = require_model()
    output = ROOT / args.output
    if args.epochs <= 0 or args.max_steps == 0 or args.max_steps < -1 or args.max_length < 64 or args.train_samples < 0:
        p.error('Invalid training parameters')
    lock = (ROOT / '.training-pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Another training pipeline is already running')
    if args.evaluate_only:
        if not (output / 'adapter/adapter_config.json').exists() or not (output / 'training-result.json').exists():
            p.error('Completed adapter and training-result.json required for evaluation-only')
    elif not args.resume and output.exists() and any(output.iterdir()):
        p.error('Training output is nonempty. Use --resume, --output or --evaluate-only')
    if args.resume and not (Path(args.resume) / 'trainer_state.json').exists():
        p.error('Resume path must contain trainer_state.json')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / 'reports' / stamp
    folder.mkdir(parents=True)
    # Keep the exact suite alongside results, even if the source is edited later.
    (folder / 'prompts.json').write_bytes((ROOT / 'evaluation_prompts.json').read_bytes())
    write_json(folder / 'model-manifest.json', manifest)
    start = time.monotonic()
    state = {'stage': 'starting', 'model_id': MODEL_ID, 'report_directory': str(folder), 'output': str(output),
             'started_utc': stamp, 'arguments': vars(args)}
    def update(stage, **extra):
        state.update(stage=stage, elapsed_seconds=time.monotonic() - start, **extra)
        write_json(ROOT / 'pipeline-status.json', state)
        write_json(folder / 'pipeline-status.json', state)
        print(f'\n=== {stage} | elapsed {state["elapsed_seconds"]:.0f}s ===', flush=True)
    def run(script, *arguments):
        subprocess.run([sys.executable, '-u', str(ROOT / script), *map(str, arguments)],
                       cwd=ROOT, check=True)
    try:
        update('baseline_evaluation')
        run('evaluate_literature.py', '--phase', 'base', '--destination', folder / 'base.json')
        if not args.evaluate_only:
            update('training')
            command = ['--epochs', args.epochs, '--max-steps', args.max_steps,
                       '--max-length', args.max_length, '--train-samples', args.train_samples, '--output', output]
            if args.resume:
                command += ['--resume', args.resume]
            run('train_literature.py', *command)
        training = read_json(output / 'training-result.json')
        if training['status'] != 'completed':
            raise RuntimeError('Training has not completed successfully')
        update('adapter_evaluation')
        run('evaluate_literature.py', '--phase', 'tuned', '--destination', folder / 'tuned.json',
            '--adapter', output / 'adapter')
        update('reporting')
        render_report(folder, training, time.monotonic() - start)
        update('completed', report=str(folder / 'report.html'), training_seconds=training['training_seconds'])
    except KeyboardInterrupt:
        update('interrupted')
        raise
    except Exception as exc:
        update('failed', error=str(exc))
        raise
    finally:
        lock.close()

if __name__ == '__main__':
    main()
