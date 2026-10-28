"""QLoRA training with live progress. Run only when ready to train."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import json
import time
from pathlib import Path
from training_progress import LiveProgress, make_progress_callback
from model_settings import MODEL_PATH, MODEL_ID, OUTPUT, MAX_LENGTH, TRAIN_SAMPLES, require_model

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--train-samples', type=int, default=TRAIN_SAMPLES)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--resume', default=None, help='Path to a Trainer checkpoint')
    parser.add_argument('--prepared-data')
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--eval-every',type=int,default=0)
    parser.add_argument('--select-best',action='store_true')
    args = parser.parse_args()
    manifest = require_model()
    if args.max_length < 64 or args.epochs <= 0 or args.max_steps == 0 or args.max_steps < -1 or args.train_samples < 0:
        parser.error('Invalid length, epochs or steps')
    output = ROOT / args.output
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error('Output is nonempty: choose --output or explicitly --resume a checkpoint')
    progress = LiveProgress(ROOT / 'training-status.json', console=True)
    training_started = None
    training_seconds = None
    try:
        from unsloth import FastModel
        import torch
        import transformers.modeling_utils as modeling_utils
        from transformers import DataCollatorForSeq2Seq
        from transformers.trainer_callback import PrinterCallback
        from trl import SFTConfig, SFTTrainer
        from prepare_literature import prepare
        if args.prepared_data:
            from datasets import load_from_disk
            prepared=ROOT/args.prepared_data
            data=load_from_disk(str(prepared))
            preparation=json.loads((prepared/'preparation.json').read_text(encoding='utf-8'))
        else:
            data, preparation = prepare(ROOT, args.max_length, args.train_samples)
        progress.update(stage='loading_model', preparation=preparation)
        original_open = modeling_utils.safe_open
        def checkpoint_open(filename, *a, **kw):
            if str(filename).startswith(str(MODEL_PATH)):
                kw['backend'] = 'pread'
            return original_open(filename, *a, **kw)
        modeling_utils.safe_open = checkpoint_open
        model, processor = FastModel.from_pretrained(
            model_name=str(MODEL_PATH), max_seq_length=args.max_length,
            dtype=torch.bfloat16, load_in_4bit=True, device_map={'': 0},
            offload_embedding=False, text_only=True, local_files_only=True,
            use_gradient_checkpointing='unsloth')
        model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0,
            finetune_vision_layers=False, finetune_audio_layers=False,
            finetune_language_layers=True, finetune_attention_modules=True,
            finetune_mlp_modules=True, use_gradient_checkpointing='unsloth')
        model.config.use_cache = False
        tokenizer = getattr(processor, 'tokenizer', processor)
        config = SFTConfig(output_dir=str(output), max_steps=args.max_steps,
            num_train_epochs=args.epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, learning_rate=args.learning_rate,
            bf16=True, optim='paged_adamw_8bit', gradient_checkpointing=True,
            logging_steps=1, save_steps=args.eval_every or 25, save_total_limit=4 if args.eval_every else 2,
            eval_strategy='steps' if args.eval_every else 'no', eval_steps=args.eval_every or 500,
            per_device_eval_batch_size=1, prediction_loss_only=True,
            load_best_model_at_end=args.select_best,metric_for_best_model='eval_loss',greater_is_better=False,
            report_to='none', disable_tqdm=True, max_length=args.max_length,
            dataset_kwargs={'skip_prepare_dataset': True}, seed=3407,
            dataloader_num_workers=0)
        trainer = SFTTrainer(model=model, processing_class=tokenizer, args=config,
            train_dataset=data['train'], eval_dataset=data['validation'] if args.eval_every else None,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100,
                                              pad_to_multiple_of=8),
            callbacks=[make_progress_callback(progress)])
        trainer.remove_callback(PrinterCallback)
        initial_validation=trainer.evaluate() if args.eval_every else None
        training_started = time.monotonic()
        result = trainer.train(resume_from_checkpoint=args.resume)
        training_seconds = time.monotonic() - training_started
        progress.update(stage='validating' if args.eval_every else 'saving')
        final_validation=trainer.evaluate() if args.eval_every else None
        progress.update(stage='saving')
        trainer.save_model(str(output / 'adapter'))
        tokenizer.save_pretrained(str(output / 'adapter'))
        (output / 'training-result.json').write_text(json.dumps({
            'status': 'completed', 'model_id': MODEL_ID, 'model_revision': manifest['revision'], 'preparation': preparation,
            'training_seconds': training_seconds,
            'total_seconds': progress.snapshot()['total_elapsed_seconds'],
            'global_step': trainer.state.global_step, 'epoch': trainer.state.epoch,
            'requested_epochs': args.epochs, 'max_steps': args.max_steps,
            'resume_from': args.resume, 'metrics': result.metrics,
            'learning_rate':args.learning_rate,'initial_validation':initial_validation,'final_validation':final_validation,
            'best_model_checkpoint':trainer.state.best_model_checkpoint,
            'timing_scope': 'This invocation of trainer.train; includes intermediate validation/checkpoint saving, excludes loading, initial/final validation and final adapter saving.'
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        progress.update(stage='completed', adapter_path=str(output / 'adapter'))
    except KeyboardInterrupt:
        progress.update(stage='interrupted')
        raise
    except Exception as exc:
        progress.update(stage='failed', error=str(exc))
        raise
    finally:
        progress.close()

if __name__ == '__main__':
    main()
