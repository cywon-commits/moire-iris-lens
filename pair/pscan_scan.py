"""Run pscan jobs in parallel (resumable). usage: python pscan_scan.py JOBFILE NPROC
Job lines are pscan_run.py arguments ('ALPHA CABS PHIM PLACE INIT [RHMODE]' or 'bg LP ALPHA'); '#' starts a comment.
Re-run the same command after any interruption: finished jobs are skipped, running ones resume from their checkpoint."""
import sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
jobs=[l.split('#')[0].split() for l in open(sys.argv[1])]; jobs=[j for j in jobs if j]
nproc=int(sys.argv[2]) if len(sys.argv)>2 else 2
here=os.path.dirname(os.path.abspath(__file__))
def go(j):
    log=open(os.path.join(here,'log_ps_'+'_'.join(j)+'.txt'),'a')
    return subprocess.call([sys.executable,os.path.join(here,'pscan_run.py')]+j,cwd=here,stdout=log,stderr=subprocess.STDOUT,
                           env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
with ThreadPoolExecutor(nproc) as ex:
    for j,rc in zip(jobs,ex.map(go,jobs)): print(' '.join(j),'exit',rc,flush=True)
