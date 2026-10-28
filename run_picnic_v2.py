import fcntl,sys,subprocess,datetime,time,traceback
from model_settings import ROOT,require_model
from picnic_evaluation import save,load
from picnic_report_v2 import report
lock=(ROOT/'.training-pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
require_model();output=ROOT/'outputs/qwen-picnic-grounded-v2'
if output.exists() and any(output.iterdir()):raise RuntimeError('Refusing nonempty output')
folder=ROOT/'reports'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-picnic-v2')
folder.mkdir();started=time.monotonic()
def status(stage,**extra):
    data={'stage':stage,'output':str(output),'report_directory':str(folder),'elapsed_seconds':time.monotonic()-started,**extra}
    save(ROOT/'pipeline-status.json',data);save(folder/'pipeline-status.json',data);print('\n=== '+stage+' ===',flush=True)
def run(script,*args):subprocess.run([sys.executable,'-u',str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)
try:
    status('prepare_retrieval_v2');run('prepare_picnic_v2.py','--freeze',folder)
    # Fail before GPU evaluations if any query would silently overflow the context.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-1.7B'),local_files_only=True)
    sizes=[len(tok(tok.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)['input_ids']) for c in load(folder/'suite.json')]
    assert max(sizes)+384<=4096,('Prompt overflow',max(sizes))
    save(folder/'preflight.json',{'cases':len(sizes),'max_prompt_tokens':max(sizes),'generation_budget':384})
    for phase,adapter in [('base',None),('previous',ROOT/'outputs/qwen-picnic-grounded-v1/adapter')]:
        status(phase+'_new_retrieval');run('picnic_evaluation.py','--folder',folder,'--phase',phase,*(['--adapter',adapter] if adapter else []))
    status('training_v2')
    run('train_literature.py','--prepared-data','data/picnic-grounded-tokenized-v2','--max-length',4096,'--epochs',4,'--learning-rate',5e-5,'--eval-every',22,'--select-best','--output',output)
    training=load(output/'training-result.json');status('evaluate_v2');run('picnic_evaluation.py','--folder',folder,'--phase','tuned','--adapter',output/'adapter')
    status('reporting');report(folder,training,time.monotonic()-started)
    # Keep previous active configuration available, publish the completed candidate separately.
    save(ROOT/'picnic-v2.json',{'adapter':str(output/'adapter'),'report':str(folder/'report.html'),'retriever':'hybrid-v2','best_checkpoint':training['best_model_checkpoint']})
    status('completed',report=str(folder/'report.html'),training_seconds=training['training_seconds'])
except BaseException as exc:
    (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8');status('failed',error=str(exc));raise
finally:lock.close()
