"""Bound QLoRA context capacity with real tokens and optimizer steps."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--lengths', default='2048,4096,8192,16384,32768,65536')
args = parser.parse_args()
lengths = [int(x) for x in args.lengths.split(',')]
report_path = ROOT / ('context-probe-' + '-'.join(map(str, lengths)) + '.json')
report = {'status': 'starting', 'lengths': lengths, 'batch_size': 1, 'rank': 8,
          'microbatches_per_optimizer_step': 2, 'optimizer_steps_per_length': 2,
          'results': []}
def save():
    report_path.write_text(json.dumps(report, indent=2, default=str))
def say(text):
    print(text, flush=True)

try:
    from unsloth import FastModel
    import torch
    import bitsandbytes as bnb
    import transformers.modeling_utils as mu
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'models/gemma-4-12B-it'), local_files_only=True)
    stats_file = ROOT/'data/literature-token-lengths.json'
    longest_file = ROOT/'data/longest-literature-tokens.json'
    if not stats_file.exists():
        all_lengths, longest = [], None
        per_split = {}
        for split in ('train', 'validation', 'test'):
            sizes = []
            with (ROOT/f'data/russian_classic_literature/{split}_dataset.json').open() as stream:
                for line in stream:
                    messages = json.loads(line)['messages']
                    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                    sizes.append(len(ids))
                    if split == 'train' and (longest is None or len(ids) > len(longest['ids'])):
                        prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=False)
                        longest = {'ids': ids, 'prompt_tokens': len(prompt_ids)}
            per_split[split] = sizes
            all_lengths.extend(sizes)
            say(f'Tokenized {split}: {len(sizes)} records')
        ordered = sorted(all_lengths)
        stats = {'count': len(ordered), 'min': ordered[0], 'max': ordered[-1],
                 'p50': ordered[int(len(ordered)*.5)], 'p90': ordered[int(len(ordered)*.9)],
                 'p95': ordered[int(len(ordered)*.95)], 'p99': ordered[int(len(ordered)*.99)],
                 'coverage': {str(n): sum(x <= n for x in ordered)/len(ordered) for n in [2048,4096,8192,16384,32768,65536]},
                 'lengths_by_split': per_split}
        stats_file.write_text(json.dumps(stats))
        longest_file.write_text(json.dumps(longest))
    stats = json.loads(stats_file.read_text())
    source = json.loads(longest_file.read_text())
    report['dataset_summary'] = {k:v for k,v in stats.items() if k != 'lengths_by_split'}
    say('Dataset lengths: '+json.dumps(report['dataset_summary']))
    original_open = mu.safe_open
    def pread(filename, *a, **kw):
        if str(filename).startswith(str(ROOT/'models/gemma-4-12B-it')):
            kw['backend'] = 'pread'
        return original_open(filename, *a, **kw)
    mu.safe_open = pread
    report['status'] = 'loading'
    save()
    model, _ = FastModel.from_pretrained(model_name=str(ROOT/'models/gemma-4-12B-it'),
        max_seq_length=max(lengths), dtype=torch.bfloat16, load_in_4bit=True,
        device_map={'':0}, offload_embedding=False, text_only=True,
        use_gradient_checkpointing='unsloth', local_files_only=True)
    model = FastModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0,
        finetune_vision_layers=False, finetune_audio_layers=False,
        finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, use_gradient_checkpointing='unsloth')
    model.train()
    model.config.use_cache=False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=1e-5)
    for length in lengths:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        entry = {'length': length, 'status': 'running', 'steps': [],
                 'repeated_source': length > len(source['ids'])}
        report['results'].append(entry)
        report['status']='testing'
        save()
        say(f'\n=== Context {length}: 2 optimizer steps, 2 microbatches each ===')
        try:
            raw = source['ids']
            ids = (raw*((length+len(raw)-1)//len(raw)))[:length]
            inputs = torch.tensor([ids], device='cuda')
            labels = inputs.clone()
            labels[:,:source['prompt_tokens']] = -100
            mask = torch.ones_like(inputs)
            for step in range(2):
                tick=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses=[]
                for micro in range(2):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out=model(input_ids=inputs, attention_mask=mask, labels=labels)
                        loss=out.loss/2
                    assert torch.isfinite(loss), 'Non-finite loss'
                    loss.backward()
                    losses.append(loss.item()*2)
                    del out,loss
                    say(f'Context {length} | step {step+1}/2 | microbatch {micro+1}/2')
                norm=torch.nn.utils.clip_grad_norm_(params,1.0)
                assert torch.isfinite(norm) and norm.item()>0, 'Invalid gradient'
                optimizer.step()
                torch.cuda.synchronize()
                entry['steps'].append({'seconds':time.perf_counter()-tick, 'loss':sum(losses)/2})
                entry.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                             free_gib=torch.cuda.mem_get_info()[0]/2**30)
                save()
                say(json.dumps(entry))
            entry['status']='passed'
            del inputs,labels,mask
        except torch.OutOfMemoryError as exc:
            entry.update(status='oom',error=str(exc),
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            save()
            say(f'Context {length}: CUDA OUT OF MEMORY')
            break
        save()
    report['status']='finished'
    save()
    say('PROBE FINISHED: '+str(report_path))
except Exception:
    report['status']='failed'
    report['error']=traceback.format_exc()
    save()
    raise
