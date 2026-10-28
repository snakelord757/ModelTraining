"""Fresh 500-example run, held-out loss, three adapter comparisons."""
import datetime,fcntl,subprocess,sys,time,shutil,html,json,traceback
from model_settings import ROOT,require_model
from evaluate_literature import read_json,write_json,render_report

output=ROOT/'outputs/qwen-selected-500-v1'
lock=(ROOT/'.training-pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if output.exists() and any(output.iterdir()): raise RuntimeError('Experiment output already exists; refusing overwrite')
require_model()
selection=read_json(ROOT/'data/qwen-selected-500-v1/preparation.json')
assert selection['splits']['train']['rows']==500 and selection['splits']['validation']['rows']==32
folder=ROOT/'reports'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ-small500')
folder.mkdir(parents=True)
shutil.copyfile(ROOT/'evaluation_prompts.json',folder/'prompts.json')
write_json(folder/'selection.json',selection)
started=time.monotonic()
def status(stage,**extra):
    state={'stage':stage,'model_id':'Qwen/Qwen3-1.7B','output':str(output),
           'report_directory':str(folder),'elapsed_seconds':time.monotonic()-started,**extra}
    write_json(ROOT/'pipeline-status.json',state);write_json(folder/'pipeline-status.json',state)
    print(f'\n=== {stage} ===',flush=True)
def run(script,*args):
    subprocess.run([sys.executable,'-u',str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)
try:
    status('baseline_evaluation')
    run('evaluate_literature.py','--phase','base','--destination',folder/'base.json')
    status('training_500_lr_2e-5')
    run('train_literature.py','--prepared-data','data/qwen-selected-500-v1',
        '--max-length',6000,'--train-samples',500,'--epochs',1,'--learning-rate',2e-5,
        '--eval-every',20,'--output',output)
    training=read_json(output/'training-result.json')
    summaries=[]
    for name,adapter,step in [('step-20',output/'checkpoint-20',20),('step-40',output/'checkpoint-40',40),('final',output/'adapter',training['global_step'])]:
        status('evaluating_'+name)
        sub=folder/name;sub.mkdir()
        shutil.copyfile(folder/'prompts.json',sub/'prompts.json')
        shutil.copyfile(folder/'base.json',sub/'base.json')
        run('evaluate_literature.py','--phase','tuned','--destination',sub/'tuned.json','--adapter',adapter)
        stage_info=dict(training)
        stage_info['global_step']=step
        stage_info['epoch']=min(step*8/500,1.0)
        stage_info['timing_note']='Training time is the entire experiment, not time to this checkpoint.'
        comparisons=render_report(sub,stage_info,time.monotonic()-started)
        summaries.append({'name':name,'step':step,'regressions':[r['id'] for r in comparisons if r['lost']],
            'new_passes':[r['id'] for r in comparisons if r['gained']],
            'repetition_failures':[r['id'] for r in comparisons if not r['tuned_checks'].get('Нет выраженного зацикливания',True)]})
    trainer_state=read_json(output/f'checkpoint-{training["global_step"]}'/'trainer_state.json')
    validation=[x for x in trainer_state['log_history'] if 'eval_loss' in x]
    write_json(folder/'experiment-summary.json',{'training':training,'validation':validation,'comparisons':summaries})
    esc=lambda v:html.escape(str(v))
    body=['<!doctype html><meta charset="utf-8"><title>500-example experiment</title><h1>Эксперимент: 500 примеров</h1>',
      '<p>Исходная Qwen, LR 2e-5, одна эпоха. Отбор эвристический; литературное качество требует прочтения ответов.</p>',
      f'<p>Время обучения с промежуточной валидацией: {training["training_seconds"]:.0f} секунд.</p>',
      '<p>Во вложенных отчётах время относится ко всему эксперименту, а не к отдельной контрольной точке.</p>',
      '<p>Сравнение на 20 запросах — диагностика для выбора настроек, не независимый итоговый тест. Параметры генерации отличаются от старого отчёта; база пересчитана.</p>']
    for s in summaries:
        body.append(f'<h2><a href="{s["name"]}/report.html">Шаг {s["step"]}</a></h2><p>Ухудшения проверок: {esc(s["regressions"])}; зацикливания: {esc(s["repetition_failures"])}</p>')
    body.append('<h2>Валидация</h2><pre>'+esc(json.dumps({'before':training['initial_validation'],'during':validation,'after':training['final_validation']},ensure_ascii=False,indent=2))+'</pre>')
    (folder/'report.html').write_text('\n'.join(body),encoding='utf-8')
    status('completed',report=str(folder/'report.html'))
except BaseException as exc:
    (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
    raise
finally:
    lock.close()
