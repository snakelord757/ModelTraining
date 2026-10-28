import fcntl,sys,subprocess,datetime,time,traceback
from model_settings import ROOT,require_model
from picnic_evaluation import prepare_suite,save,load,report
lock=(ROOT/'.training-pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
require_model()
output=ROOT/'outputs/qwen-picnic-grounded-v1'
if output.exists() and any(output.iterdir()):raise RuntimeError('Output exists; refusing overwrite')
folder=ROOT/'reports'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-picnic')
folder.mkdir(parents=True);started=time.monotonic()
def status(stage,**extra):
    data={'stage':stage,'model_id':'Qwen/Qwen3-1.7B','output':str(output),'report_directory':str(folder),'elapsed_seconds':time.monotonic()-started,**extra}
    save(ROOT/'pipeline-status.json',data);save(folder/'pipeline-status.json',data);print('\n=== '+stage+' ===',flush=True)
def run(script,*args):subprocess.run([sys.executable,'-u',str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)
try:
    status('freezing_retrieval_context');prepare_suite(folder)
    status('baseline_with_retrieval');run('picnic_evaluation.py','--folder',folder,'--phase','base')
    status('grounded_training')
    run('train_literature.py','--prepared-data','data/picnic-grounded-tokenized-v1','--max-length',4096,
        '--epochs',3,'--learning-rate',2e-5,'--eval-every',12,'--select-best','--output',output)
    training=load(output/'training-result.json')
    status('adapter_with_same_retrieval');run('picnic_evaluation.py','--folder',folder,'--phase','tuned','--adapter',output/'adapter')
    status('reporting');report(folder,training,time.monotonic()-started)
    save(ROOT/'picnic-active.json',{'adapter':str(output/'adapter'),'report':str(folder/'report.html'),'source':'user PDF','best_checkpoint':training['best_model_checkpoint']})
    status('completed',report=str(folder/'report.html'),training_seconds=training['training_seconds'])
except BaseException as exc:
    (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8');status('failed',error=str(exc));raise
finally:lock.close()
