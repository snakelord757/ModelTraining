"""Live, atomically written training status (no torch dependency)."""
import json
import threading
import time
from pathlib import Path

class LiveProgress:
    def __init__(self, path, interval=1, console=False, console_stream=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.training_started = None
        self.training_finished = None
        self.data = {'stage': 'preparing', 'step': 0, 'total_steps': None, 'loss': None}
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.stop = threading.Event()
        self.interval = interval
        self.console_bar = None
        if console:
            from tqdm.auto import tqdm
            self.console_bar = tqdm(total=100, file=console_stream, dynamic_ncols=True,
                bar_format='{desc} |{bar}| {n:6.2f}% | {postfix}', leave=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update(self, **values):
        with self.lock:
            if values.get('stage') == 'training' and self.training_started is None:
                self.training_started = time.monotonic()
            if values.get('stage') in ('validating', 'saving', 'completed', 'failed', 'interrupted') and self.training_started is not None and self.training_finished is None:
                self.training_finished = time.monotonic()
            self.data.update(values)
        self.write()

    def snapshot(self):
        with self.lock:
            data = dict(self.data)
            now = time.monotonic()
            elapsed = (self.training_finished or now) - self.training_started if self.training_started is not None else 0
            total, step = data['total_steps'], data['step']
            new_steps = step - data.get('resume_start_step', 0)
            data.update(total_elapsed_seconds=now-self.started,
                        training_elapsed_seconds=elapsed, updated_at=time.time(),
                        percent=100*step/total if total else None,
                        eta_seconds=elapsed/new_steps*(total-step) if total and new_steps > 0 else None)
            return data

    def write(self):
        # Serialize writers so the heartbeat and callbacks never share a temp file.
        with self.write_lock:
            tmp = self.path.with_suffix('.tmp')
            state = self.snapshot()
            tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
            tmp.replace(self.path)
            if self.console_bar is not None:
                self.console_bar.n = state['percent'] or 0
                self.console_bar.set_description_str(state['stage'], refresh=False)
                self.console_bar.set_postfix_str(
                    f"{state['step']}/{state['total_steps'] or '?'} steps | "
                    f"Training {format_duration(state['training_elapsed_seconds'])} | "
                    f"ETA {format_duration(state['eta_seconds'])} | "
                    f"Total {format_duration(state['total_elapsed_seconds'])} | "
                    f"Loss {state['loss'] if state['loss'] is not None else '-'}", refresh=False)
                self.console_bar.refresh()

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.write()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.write()
        if self.console_bar is not None:
            self.console_bar.close()

def format_duration(seconds):
    if seconds is None:
        return '--:--:--'
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

def make_progress_callback(progress):
    from transformers import TrainerCallback
    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            progress.update(stage='training', step=state.global_step, total_steps=state.max_steps,
                            resume_start_step=state.global_step)
        def on_step_end(self, args, state, control, **kwargs):
            progress.update(step=state.global_step, total_steps=state.max_steps)
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                progress.update(loss=logs['loss'])
    return ProgressCallback()

