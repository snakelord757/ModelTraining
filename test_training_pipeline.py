"""Exercise the real TRL/collator/progress integration using a tiny test model."""
import json
import tempfile
from pathlib import Path
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, GPT2Config, GPT2LMHeadModel
from trl import SFTConfig, SFTTrainer
from training_progress import LiveProgress, make_progress_callback
from model_settings import MODEL_PATH

root = Path(__file__).resolve().parent
data = load_from_disk(str(root / 'data/prepared-qwen3-1.7b-whole-4096-1000'))
assert 0 < len(data['train']) <= 1000
for split in data.values():
    for row in split:
        assert len(row['input_ids']) == len(row['labels']) <= 4096
        assert -100 in row['labels'] and any(x != -100 for x in row['labels'][1:])
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
ids = [x for x in data['train'][0]['labels'] if x != -100][:16]
mini = Dataset.from_list([{'input_ids': ids, 'attention_mask': [1]*len(ids),
                          'labels': [-100]*4+ids[4:]}]*2)
model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=32,
                                  n_embd=8, n_layer=1, n_head=1))
with tempfile.TemporaryDirectory() as folder:
    progress = LiveProgress(Path(folder) / 'status.json', interval=0.05)
    try:
        config = SFTConfig(output_dir=folder, max_steps=2, per_device_train_batch_size=1,
            use_cpu=False, bf16=False, fp16=False, gradient_checkpointing=False,
            logging_steps=1, save_strategy='no', report_to='none',
            eval_strategy='steps',eval_steps=1,per_device_eval_batch_size=1,prediction_loss_only=True,
            dataset_kwargs={'skip_prepare_dataset': True}, max_length=32,
            dataloader_pin_memory=False, optim='adamw_torch')
        trainer = SFTTrainer(model=model, processing_class=tokenizer, args=config,
            train_dataset=mini, eval_dataset=mini,
            data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
            callbacks=[make_progress_callback(progress)])
        initial=trainer.evaluate()
        assert 'eval_loss' in initial
        trainer.train()
        final=trainer.evaluate()
        assert 'eval_loss' in final
        state = progress.snapshot()
        assert state['step'] == state['total_steps'] == 2
        assert state['percent'] == 100 and state['loss'] is not None
        assert state['training_elapsed_seconds'] > 0
        progress.update(stage='completed')
    finally:
        progress.close()
print('PASS: dataset labels, TRL integration, 2 tiny-model GPU steps, live progress and loss')
