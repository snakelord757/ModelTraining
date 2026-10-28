"""Two optimizer steps at exactly 12288 tokens; no adapter is saved."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
import time
import traceback
from model_settings import ROOT, MODEL_PATH, MODEL_ID, require_model

report_path = ROOT / 'qwen-12k-probe.json'
report = {'model_id': MODEL_ID, 'status': 'starting', 'length': 12288,
          'batch_size': 1, 'gradient_accumulation': 8, 'steps': []}
def save():
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)

try:
    require_model()
    from unsloth import FastModel
    import torch
    import bitsandbytes as bnb
    import transformers.modeling_utils as mu
    from datasets import load_from_disk
    data = load_from_disk(str(ROOT / 'data/prepared-qwen3-1.7b-whole-12288-0'))['train']
    source = max(data, key=lambda row: len(row['input_ids']))
    length = report['length']
    repeat = (length + len(source['input_ids']) - 1) // len(source['input_ids'])
    report['repeated_source'] = repeat > 1
    original = mu.safe_open
    def pread(filename, *a, **kw):
        if str(filename).startswith(str(MODEL_PATH)):
            kw['backend'] = 'pread'
        return original(filename, *a, **kw)
    mu.safe_open = pread
    model, _ = FastModel.from_pretrained(model_name=str(MODEL_PATH), max_seq_length=length,
        dtype=torch.bfloat16, load_in_4bit=True, device_map={'': 0},
        offload_embedding=False, text_only=True, local_files_only=True,
        use_gradient_checkpointing='unsloth')
    model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0,
        finetune_vision_layers=False, finetune_audio_layers=False,
        finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, use_gradient_checkpointing='unsloth')
    model.train()
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=1e-4)
    ids = torch.tensor([(source['input_ids'] * repeat)[:length]], device='cuda')
    labels = torch.tensor([(source['labels'] * repeat)[:length]], device='cuda')
    mask = torch.ones_like(ids)
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(8):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = result.loss / 8
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward()
            losses.append(loss.item() * 8)
            del result, loss
            print(f'12k probe step {step+1}/2, microbatch {micro+1}/8', flush=True)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        assert torch.isfinite(norm) and norm.item() > 0, 'Invalid gradients'
        optimizer.step()
        torch.cuda.synchronize()
        report['steps'].append({'seconds': time.monotonic()-start, 'loss': sum(losses)/8})
        report.update(status='testing', peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      free_gib=torch.cuda.mem_get_info()[0]/2**30)
        save()
    report['status'] = 'passed'
    save()
except Exception:
    report.update(status='failed', error=traceback.format_exc())
    save()
    raise
