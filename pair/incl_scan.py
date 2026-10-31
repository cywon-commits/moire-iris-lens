"""Run a list of inclusion jobs in parallel (resumable). usage: python incl_scan.py JOBFILE NPROC
JOBFILE lines: 'ALPHA EPS V0 LP R0'  (comments with #). Re-run the same command after any interruption."""
import sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
jobs=[l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
nproc=int(sys.argv[2]) if len(sys.argv)>2 else 2
here=os.path.dirname(os.path.abspath(__file__))
def go(j):
    log=open(os.path.join(here,'log_inc_'+'_'.join(j)+'.txt'),'a')
    return subprocess.call([sys.executable,os.path.join(here,'incl_ck.py')]+j,cwd=here,stdout=log,stderr=subprocess.STDOUT,
                           env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
with ThreadPoolExecutor(nproc) as ex:
    for j,rc in zip(jobs,ex.map(go,jobs)): print(' '.join(j),'exit',rc,flush=True)
