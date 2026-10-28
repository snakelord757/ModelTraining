import json
import tempfile
import time
import unittest
import io
from pathlib import Path
from training_progress import LiveProgress

class ProgressTest(unittest.TestCase):
    def test_console_timer_and_freeze_on_saving(self):
        with tempfile.TemporaryDirectory() as folder:
            stream = io.StringIO()
            progress = LiveProgress(Path(folder)/'status.json', interval=0.02,
                                    console=True, console_stream=stream)
            try:
                progress.update(stage='training', step=1, total_steps=4)
                time.sleep(0.08)
                self.assertIn('25.00%', stream.getvalue())
                self.assertIn('Training 00:00:00', stream.getvalue())
                self.assertGreater(stream.getvalue().count('Training'), 1)
                progress.update(stage='saving', step=4)
                elapsed = progress.snapshot()['training_elapsed_seconds']
                time.sleep(0.05)
                self.assertEqual(elapsed, progress.snapshot()['training_elapsed_seconds'])
                self.assertIn('100.00%', stream.getvalue())
            finally:
                progress.close()

    def test_heartbeat_percentage_resume_and_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            progress = LiveProgress(path, interval=0.02)
            try:
                progress.update(stage='training', step=10, total_steps=20, resume_start_step=10)
                self.assertIsNone(progress.snapshot()['eta_seconds'])
                progress.update(step=12, loss=2.0)
                first = progress.snapshot()
                self.assertEqual(first['percent'], 60)
                time.sleep(0.1)
                current = json.loads(path.read_text())
                self.assertGreater(current['training_elapsed_seconds'], first['training_elapsed_seconds'])
                self.assertAlmostEqual(current['eta_seconds'], current['training_elapsed_seconds'] * 4)
                progress.update(step=20, stage='saving')
                self.assertEqual(progress.snapshot()['percent'], 100)
                self.assertEqual(progress.snapshot()['stage'], 'saving')
                progress.update(stage='completed')
            finally:
                progress.close()
            self.assertFalse(progress.thread.is_alive())
            self.assertEqual(json.loads(path.read_text())['stage'], 'completed')

if __name__ == '__main__':
    unittest.main()
