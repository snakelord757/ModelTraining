"""Local, reproducible before/after generation and bounded automatic checks."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import difflib
import hashlib
import html
import json
import re
import time
import traceback
from pathlib import Path
from model_settings import MODEL_ID, MODEL_PATH, OUTPUT, TEMPLATE_KWARGS

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / 'evaluation_prompts.json'
SETTINGS = {'max_new_tokens': 768, 'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.0}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def checks(case, answer):
    """Explicit task requirements only; no model-as-judge or aesthetic scores."""
    result = {}
    words = re.findall(r"[\w]+(?:[-’'][\w]+)*", answer, re.UNICODE)
    grams=[tuple(w.casefold() for w in words[i:i+10]) for i in range(max(0,len(words)-9))]
    result['Нет выраженного зацикливания'] = len(grams)<20 or 1-len(set(grams))/len(grams)<0.2
    if 'exact' in case:
        # Ignore case and spacing; keep punctuation to check the requested format.
        normalize = lambda s: re.sub(r'\s+', '', s).casefold()
        result['Точный ответ и формат'] = normalize(answer) == normalize(case['exact'])
    if 'json_value' in case:
        try:
            result['Корректный JSON и значения'] = json.loads(answer) == case['json_value']
        except (ValueError, TypeError):
            result['Корректный JSON и значения'] = False
    if 'min_words' in case:
        result['Заданный диапазон слов'] = case['min_words'] <= len(words) <= case['max_words']
    if 'paragraphs' in case:
        result['Число абзацев'] = len(re.split(r'\n\s*\n', answer.strip())) == case['paragraphs']
    if 'forbidden' in case:
        result['Нет запрещённых слов'] = not (set(w.casefold() for w in words) & set(case['forbidden']))
    return result

def compare(case, base, tuned):
    a, b = checks(case, base['answer']), checks(case, tuned['answer'])
    lost = [k for k in a if a[k] and not b[k]]
    gained = [k for k in a if not a[k] and b[k]]
    stable = [f'{k}: {"выполнено" if a[k] else "не выполнено"}' for k in a if a[k] == b[k]]
    return {'id': case['id'], 'lost': lost, 'gained': gained, 'stable': stable,
            'identical': base['answer'] == tuned['answer'],
            'similarity': difflib.SequenceMatcher(None, base['answer'], tuned['answer'], autojunk=False).ratio(),
            'base_checks': a, 'tuned_checks': b,
            'truncated': base['hit_token_limit'] or tuned['hit_token_limit']}

def generate(phase, destination, adapter):
    from unsloth import FastModel
    import torch
    import transformers.modeling_utils as modeling_utils
    from transformers import set_seed
    from tqdm.auto import tqdm

    suite = destination.parent / 'prompts.json'
    cases = read_json(suite)
    assert len(cases) == 20 and len({c['id'] for c in cases}) == 20
    original_open = modeling_utils.safe_open
    def checkpoint_open(filename, *a, **kw):
        if str(filename).startswith(str(MODEL_PATH)):
            kw['backend'] = 'pread'
        return original_open(filename, *a, **kw)
    modeling_utils.safe_open = checkpoint_open
    model, processor = FastModel.from_pretrained(
        model_name=str(MODEL_PATH), max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=True, device_map={'': 0},
        offload_embedding=False, text_only=True, local_files_only=True)
    if phase == 'tuned':
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    FastModel.for_inference(model)
    model.eval()
    model.config.use_cache = True
    tokenizer = getattr(processor, 'tokenizer', processor)
    set_seed(3407)
    document = {'phase': phase, 'suite_sha256': hashlib.sha256(suite.read_bytes()).hexdigest(),
                'settings': SETTINGS, 'seed': 3407, 'quantization': '4-bit, same loader for both models',
                'model_id': MODEL_ID, 'model_path': str(MODEL_PATH), 'template_kwargs': TEMPLATE_KWARGS,
                'adapter': str(adapter) if phase == 'tuned' else None, 'answers': []}
    start = time.monotonic()
    for case in tqdm(cases, desc=f'Evaluation {phase}', unit='prompt'):
        set_seed(3407)
        messages = [{'role': 'user', 'content': case['prompt']}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
        if inputs['input_ids'].shape[-1] + SETTINGS['max_new_tokens'] > 2048:
            raise ValueError('Evaluation prompt exceeds context')
        begin = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, **SETTINGS, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id)
        tokens = output[0, inputs['input_ids'].shape[-1]:].tolist()
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        document['answers'].append({'id': case['id'], 'prompt': case['prompt'],
            'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            'tokens': len(tokens), 'seconds': time.monotonic() - begin,
            'hit_token_limit': len(tokens) >= SETTINGS['max_new_tokens'] and tokens[-1] not in eos})
        document['generation_seconds'] = time.monotonic() - start
        write_json(destination, document)

def render_report(folder, training, pipeline_seconds):
    folder = Path(folder)
    suite = folder / 'prompts.json'
    cases = read_json(suite)
    base, tuned = read_json(folder / 'base.json'), read_json(folder / 'tuned.json')
    digest = hashlib.sha256(suite.read_bytes()).hexdigest()
    if base['suite_sha256'] != digest or tuned['suite_sha256'] != digest or base['settings'] != tuned['settings']:
        raise ValueError('Cannot compare different suites or generation settings')
    if base.get('model_id') != tuned.get('model_id') or base.get('template_kwargs') != tuned.get('template_kwargs'):
        raise ValueError('Cannot compare different base models or chat settings')
    ids = [c['id'] for c in cases]
    if [a['id'] for a in base['answers']] != ids or [a['id'] for a in tuned['answers']] != ids:
        raise ValueError('Need all 20 answers in both phases')
    results = [compare(c, a, b) for c, a, b in zip(cases, base['answers'], tuned['answers'])]
    regressions = [r['id'] for r in results if r['lost']]
    from training_progress import format_duration
    training_seconds = training.get('training_seconds')
    timings = {'training_seconds': training_seconds,
               'training_setup_and_save_seconds': training.get('total_seconds'),
               'generation_seconds': base['generation_seconds'] + tuned['generation_seconds'],
               'pipeline_seconds_before_report': pipeline_seconds}
    write_json(folder / 'comparison.json', {'timings': timings, 'training': training,
        'regressions_on_checked_requirements': regressions, 'comparisons': results,
        'scope': 'Automatic explicit constraint checks only. Literary quality requires human review.'})
    esc = lambda x: html.escape(str(x))
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение моделей</title>',
        '<style>body{font:17px system-ui;max-width:1400px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{vertical-align:top;border:1px solid #ccc;padding:12px}.bad{color:#b42318}details{margin:16px 0}h2{margin-top:40px}</style>',
        f'<h1>{esc(base.get("model_id", "Model"))}: базовая модель и литературный адаптер</h1>',
        f'<p>{esc(training.get("timing_note", ""))}</p>',
        f'<p>Подготовка данных: {esc(json.dumps(training.get("preparation", {}), ensure_ascii=False))}</p>',
        f'<p>Время обучения: <b>{format_duration(training_seconds)}</b>. '
        f'Обучение с загрузкой и сохранением: {format_duration(training.get("total_seconds"))}. '
        f'Генерация 40 ответов: {format_duration(timings["generation_seconds"])}. '
        f'Весь запуск до формирования отчёта: {format_duration(pipeline_seconds)}.</p>',
        f'<p>Шагов: {esc(training.get("global_step"))}; эпоха: {esc(training.get("epoch"))}; '
        f'ограничение шагов: {esc(training.get("max_steps"))} (-1 означает отсутствие ограничения).</p>',
        '<p>Время обучения включает сохранение промежуточных контрольных точек. При возобновлении указано время только текущего запуска.</p>',
        '<p>20 новых фиксированных запросов, не используемых для обучения; одинаковые шаблон чата, '
        f'4-битная загрузка и одинаковые параметры генерации: {esc(json.dumps(base["settings"]))}. '
        'Это небольшая диагностическая выборка, а не общая оценка качества.</p>',
        '<p>Автопроверки оценивают только явно перечисленные условия. Художественная выразительность, '
        'связность и сохранение смысла требуют чтения ответов. Совпадение текста не равно оценке качества. '
        'Ответы, достигшие лимита генерации, помечены отдельно.</p>',
        f'<p class="bad">Потеря ранее выполненных условий: {esc(", ".join(regressions) or "не обнаружена на проверяемых условиях; это не доказывает отсутствие ухудшений")}</p>']
    for c, a, b, r in zip(cases, base['answers'], tuned['answers'], results):
        changed = []
        if r['identical']:
            changed.append('Ответы полностью совпадают.')
        else:
            changed.append(f'Ответ изменился; сходство символов {r["similarity"]:.1%}.')
        changed.append(f'Длина: {a["tokens"]} → {b["tokens"]} токенов.')
        changed.append('Утраченные условия: ' + (', '.join(r['lost']) or 'нет'))
        changed.append('Новые выполненные условия: ' + (', '.join(r['gained']) or 'нет'))
        stable = '; '.join(r['stable']) or 'Нет одинаковых результатов автоматических проверок.'
        parts += [f'<h2 id="{c["id"]}">{c["id"]} — {esc(c["category"])}</h2>',
            f'<p><b>Запрос:</b> {esc(c["prompt"])}</p>',
            f'<p><b>Что изменилось:</b> {esc("; ".join(changed))}</p>',
            f'<p><b>Что осталось прежним:</b> {esc(stable)}</p>',
            '<p><b>Качественная оценка:</b> требует ручной проверки смысла, связности и стиля.</p>' if c['category'] == 'Литература' else '',
            '<p class="bad">Внимание: хотя бы один ответ достиг лимита токенов; сравните незавершённость отдельно.</p>' if r['truncated'] else '',
            '<table><tr><th>Базовая модель</th><th>Дообученная модель</th></tr>',
            f'<tr><td><pre>{esc(a["answer"])}</pre></td><td><pre>{esc(b["answer"])}</pre></td></tr></table>',
            '<details><summary>Построчные изменения (− база, + адаптер)</summary><pre>' +
            esc('\n'.join(difflib.unified_diff(a['answer'].splitlines(), b['answer'].splitlines(), lineterm=''))) + '</pre></details>']
    parts.append('</html>')
    (folder / 'report.html').write_text('\n'.join(parts), encoding='utf-8')
    print(f'\nReport: {folder / "report.html"}\nTraining time: {format_duration(training_seconds)}\n'
          f'Regressions on checked requirements: {", ".join(regressions) or "none detected"}', flush=True)
    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['base', 'tuned'], required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--adapter', type=Path, default=ROOT / OUTPUT / 'adapter')
    args = p.parse_args()
    try:
        generate(args.phase, args.destination, args.adapter)
    except Exception:
        (args.destination.parent / f'{args.phase}-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
