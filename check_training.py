"""Check CUDA, NF4 backward, and training imports without model downloads."""
import importlib.metadata as metadata
import json
import unsloth
import torch

assert torch.cuda.is_available(), 'CUDA unavailable'
device = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
(x @ x.T).float().square().mean().backward()
torch.cuda.synchronize()
import bitsandbytes as bnb
layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                         quant_type='nf4', compress_statistics=True).cuda()
z = torch.randn(2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
layer(z).float().square().mean().backward()
assert z.grad is not None and torch.isfinite(z.grad).all()
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
assert 'gemma4_unified' in CONFIG_MAPPING, 'Transformers lacks Gemma 4 12B architecture'
from peft import LoraConfig
from trl import SFTTrainer
print(json.dumps({
    'gpu': device, 'capability': capability, 'cuda': torch.version.cuda,
    'bf16': torch.cuda.is_bf16_supported(),
    'versions': {name: metadata.version(name) for name in
                 ['torch', 'transformers', 'peft', 'trl', 'datasets', 'accelerate', 'bitsandbytes', 'unsloth']},
    'cuda_matmul_backward': 'PASS', 'nf4_backward': 'PASS',
    'gemma4_unified_config': 'PASS',
}, indent=2))
