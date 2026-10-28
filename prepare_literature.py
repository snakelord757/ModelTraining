"""Validate and tokenize local JSONL conversations, preserving source splits."""
import json
from pathlib import Path
from model_settings import MODEL_PATH, MODEL_ID, MAX_LENGTH, TRAIN_SAMPLES, TEMPLATE_KWARGS

def prepare(root, max_length=MAX_LENGTH, train_samples=TRAIN_SAMPLES):
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict, load_from_disk
    root = Path(root)
    source = root / 'data/russian_classic_literature'
    output = root / f'data/prepared-qwen3-1.7b-whole-{max_length}-{train_samples}'
    if (output / 'preparation.json').exists():
        return load_from_disk(str(output)), json.loads((output / 'preparation.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    result, stats = {}, {}
    for split in ('train', 'validation', 'test'):
        records = []
        counts = dict(source_rows=0, kept_rows=0, truncated_rows=0, skipped_long_rows=0, no_target_rows=0, invalid_rows=0, target_tokens=0, max_source_tokens=0)
        with (source / f'{split}_dataset.json').open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                counts['source_rows'] += 1
                item = json.loads(line)
                messages = item.get('messages', [])
                if not messages or messages[-1].get('role') != 'assistant' or any(not isinstance(m.get('content'), str) for m in messages):
                    counts['invalid_rows'] += 1
                    continue
                full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **TEMPLATE_KWARGS)
                prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False, **TEMPLATE_KWARGS)
                if not full.startswith(prompt):
                    raise ValueError('Chat template does not preserve prompt prefix')
                encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
                counts['max_source_tokens'] = max(counts['max_source_tokens'], len(encoded['input_ids']))
                if len(encoded['input_ids']) > max_length:
                    counts['skipped_long_rows'] += 1
                    continue
                ids = encoded['input_ids'][:max_length]
                offsets = encoded['offset_mapping'][:max_length]
                labels = [token if start >= len(prompt) and end > start else -100
                          for token, (start, end) in zip(ids, offsets)]
                if not any(label != -100 for label in labels[1:]):
                    counts['no_target_rows'] += 1
                    continue
                records.append({'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': labels})
                counts['target_tokens'] += sum(label != -100 for label in labels[1:])
                if counts['source_rows'] % 500 == 0:
                    print(f'Preparing {split}: {counts["source_rows"]} rows', flush=True)
        if not records:
            raise ValueError(f'No usable examples in {split}')
        counts['eligible_rows'] = len(records)
        if split == 'train' and train_samples > 0 and len(records) > train_samples:
            import random
            records = random.Random(3407).sample(records, train_samples)
        counts['kept_rows'] = len(records)
        counts['target_tokens'] = sum(sum(t != -100 for t in r['labels'][1:]) for r in records)
        stats[split] = counts
        result[split] = Dataset.from_list(records)
    data = DatasetDict(result)
    data.save_to_disk(str(output))
    report = {'model_id': MODEL_ID, 'max_length': max_length, 'train_samples': train_samples, 'splits': stats,
              'loss': 'Assistant turn only; prompt labels are -100',
              'truncation': 'Skip overlength examples, never truncate; seed 3407 sample; never train on validation/test'}
    (output / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return data, report

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--train-samples', type=int, default=TRAIN_SAMPLES)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parent, args.max_length, args.train_samples)
