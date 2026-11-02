"""Run pscan jobs in parallel (resumable). usage: python pscan_scan.py JOBFILE NPROC [SCRIPT]   (SCRIPT default pscan_run.py; hetero_run.py for P3)
Job lines are pscan_run.py arguments ('ALPHA CABS PHIM PLACE INIT [RHMODE]' or 'bg LP ALPHA'); '#' starts a comment.
Re-run the same command after any interruption: finished jobs are skipped, running ones resume from their checkpoint."""
import sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
jobs=[l.split('#')[0].split() for l in open(sys.argv[1])]; jobs=[j for j in jobs if j]
nproc=int(sys.argv[2]) if len(sys.argv)>2 else 2
here=os.path.dirname(os.path.abspath(__file__))
script=sys.argv[3] if len(sys.argv)>3 else 'pscan_run.py'; pre='log_ps_' if script=='pscan_run.py' else 'log_'+script.split('_')[0]+'_'
def go(j):
    log=open(os.path.join(here,pre+'_'.join(j)+'.txt'),'a')
    return subprocess.call([sys.executable,os.path.join(here,script)]+j,cwd=here,stdout=log,stderr=subprocess.STDOUT,
                           env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
with ThreadPoolExecutor(nproc) as ex:
    for j,rc in zip(jobs,ex.map(go,jobs)): print(' '.join(j),'exit',rc,flush=True)
