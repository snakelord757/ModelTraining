"""Three synthetic QLoRA steps; does not save or modify model weights."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
import time
import traceback
from pathlib import Path

ROOT = Path('/mnt/d/ModelTraining')
REPORT = ROOT / 'qlora-smoke-result.json'
result = {'status': 'starting', 'sequence_length': 512, 'batch_size': 1,
          'lora_rank': 8, 'steps_requested': 3, 'steps': []}
def save():
    REPORT.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)

save()
try:
    from unsloth import FastModel
    import torch
    import bitsandbytes as bnb
    # DrvFS mmap of the single 24-GB checkpoint exceeds this WSL commit budget.
    # The installed safetensors supports pread (also used by Transformers on Windows).
    # Scope the override to this process and this local checkpoint only.
    import transformers.modeling_utils as modeling_utils
    original_safe_open = modeling_utils.safe_open
    def checkpoint_safe_open(filename, *args, **kwargs):
        if str(filename).startswith(str(ROOT / 'models/gemma-4-12B-it')):
            kwargs['backend'] = 'pread'
        return original_safe_open(filename, *args, **kwargs)
    modeling_utils.safe_open = checkpoint_safe_open
    result['checkpoint_read_backend'] = 'pread'
    torch.manual_seed(3407)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result['status'] = 'loading'
    save()
    model, processor = FastModel.from_pretrained(
        model_name=str(ROOT / 'models/gemma-4-12B-it'),
        max_seq_length=512, dtype=torch.bfloat16, load_in_4bit=True,
        device_map={'': 0}, offload_embedding=False, text_only=True,
        use_gradient_checkpointing='unsloth', local_files_only=True,
    )
    result['load_seconds'] = time.perf_counter() - started
    result['loaded_allocated_gib'] = torch.cuda.memory_allocated() / 2**30
    model = FastModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0, bias='none',
        finetune_vision_layers=False, finetune_audio_layers=False,
        finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, use_gradient_checkpointing='unsloth',
    )
    model.train()
    model.config.use_cache = False
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n, _ in trainable)
    result['trainable_parameters'] = sum(p.numel() for _, p in trainable)
    result['parameter_devices'] = sorted(set(str(p.device) for p in model.parameters()))
    result['quantization_config'] = model.config.quantization_config.to_dict() if hasattr(model.config.quantization_config, 'to_dict') else model.config.quantization_config
    tokenizer = getattr(processor, 'tokenizer', processor)
    text = 'User: Classify the request: payment was charged twice. Assistant: Payments. '
    tokens = tokenizer.encode(text, add_special_tokens=False)
    ids = (tokens * (512 // len(tokens) + 1))[:512]
    inputs = torch.tensor([ids], device='cuda')
    labels = inputs.clone()
    labels[:, :256] = -100
    optimizer = bnb.optim.PagedAdamW8bit([p for _, p in trainable], lr=1e-4)
    result['status'] = 'training'
    save()
    for step in range(3):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels)
            loss = output.loss
        assert torch.isfinite(loss), 'Non-finite loss'
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        assert torch.isfinite(grad_norm) and grad_norm.item() > 0, 'Invalid gradient norm'
        changed = next(p for _, p in trainable if p.grad is not None and p.grad.abs().max().item() > 0)
        before = changed.detach().clone()
        optimizer.step()
        torch.cuda.synchronize()
        assert not torch.equal(before, changed.detach()), 'Optimizer did not update adapter'
        del before
        result['steps'].append({'step': step + 1, 'loss': loss.item(),
                                'gradient_norm': grad_norm.item(),
                                'seconds': time.perf_counter() - tick})
        del output, loss
        result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        result['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        save()
    result['status'] = 'passed'
    result['total_seconds'] = time.perf_counter() - started
    result['free_gpu_gib_at_end'] = torch.cuda.mem_get_info()[0] / 2**30
    save()
except Exception:
    result['status'] = 'failed'
    result['error'] = traceback.format_exc()
    if 'torch' in globals() and torch.cuda.is_available():
        result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        result['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    save()
    raise
